"""Orchestrator: ingest → detect → track → features → events → storage.

Run:
    python -m src.main --config config.yaml [--show]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from .config import load_config
from .detector import VehicleDetector
from .events import Event, EventEngine
from .features import FeatureExtractor
from .storage import Storage
from .stream import LiveStream
from .tracker import VehicleTracker

log = logging.getLogger(__name__)

_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_LABEL_ANNOTATOR = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)


def _annotate(
    frame: np.ndarray,
    tracked: sv.Detections,
    tracker: VehicleTracker,
    feats,
    detector: VehicleDetector,
) -> np.ndarray:
    out = frame.copy()

    # Counting lines + their counters.
    for line in tracker.lines:
        cv2.line(out, line.p1, line.p2, (0, 200, 255), 2)
        mid = ((line.p1[0] + line.p2[0]) // 2,
               (line.p1[1] + line.p2[1]) // 2)
        cv2.putText(
            out,
            f"{line.name}  IN:{line.zone.in_count} OUT:{line.zone.out_count}",
            (mid[0] - 80, mid[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
        )

    # Detections + track IDs.
    if len(tracked) > 0:
        labels = []
        for i in range(len(tracked)):
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            tid = int(tracked.tracker_id[i]) \
                if tracked.tracker_id is not None and tracked.tracker_id[i] is not None \
                else -1
            labels.append(f"#{tid} {detector.class_name(cid)}")
        out = _BOX_ANNOTATOR.annotate(scene=out, detections=tracked)
        out = _LABEL_ANNOTATOR.annotate(
            scene=out, detections=tracked, labels=labels,
        )

    # HUD (top-left).
    hud = [
        f"vehicles: {feats.n_vehicles}",
        f"mean_speed_px: {feats.mean_speed_px:5.2f}",
        f"density: {feats.density:.3f}",
        f"congestion: {feats.congestion:.2f}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 25 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Congestion bar (top-right).
    h, w = out.shape[:2]
    bar_w, bar_h = 220, 14
    x0, y0 = w - bar_w - 15, 15
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h),
                  (80, 80, 80), 1)
    fill = int(bar_w * float(feats.congestion))
    color = (0, 200, 0) if feats.congestion < 0.6 else (
        0, 165, 255) if feats.congestion < 0.85 else (0, 0, 255)
    cv2.rectangle(out, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.putText(out, "congestion", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return out


class Runner:
    def __init__(self, config_path: str, show: bool) -> None:
        self.cfg = load_config(config_path)
        self.show = show or self.cfg.get("display", {}).get("show_window", False)

        s = self.cfg["stream"]
        m = self.cfg["model"]
        f = self.cfg.get("features", {})
        e = self.cfg.get("events", {})
        st = self.cfg.get("storage", {})

        self.events_engine = EventEngine(
            congestion_high_threshold=e.get("congestion_high_threshold", 0.6),
            congestion_high_min_seconds=e.get("congestion_high_min_seconds", 10),
        )

        self.storage = Storage(
            sqlite_path=st.get("sqlite_path", "data/traffic.db"),
            csv_dir=st.get("csv_dir", "data"),
            rollup_seconds=st.get("rollup_seconds", 60),
        )

        self.stream = LiveStream(
            youtube_url=s["url"],
            frame_stride=s.get("frame_stride", 5),
            resize_width=s.get("resize_width", 960),
            reresolve_seconds=s.get("reresolve_seconds", 1800),
            max_read_failures=s.get("max_read_failures", 30),
            on_stream_down=self._on_stream_down,
        )

        self.detector = VehicleDetector(
            weights=m.get("weights", "yolov8n.pt"),
            conf=m.get("conf", 0.35),
            iou=m.get("iou", 0.5),
            vehicle_class_ids=m.get("vehicle_class_ids"),
            imgsz=m.get("imgsz", 640),
            device=m.get("device", ""),
        )

        self.tracker = VehicleTracker(
            lines_cfg=self.cfg.get("lines", []),
            class_name_fn=self.detector.class_name,
        )

        # Estimate stopped_min_frames from seconds * inferred fps.
        # We don't really know live fps; stride 5 against 30fps source ≈ 6 fps.
        approx_fps = max(1.0, 30.0 / max(1, s.get("frame_stride", 5)))
        self.features = FeatureExtractor(
            roi_polygon=f.get("roi_polygon") or None,
            speed_window_frames=f.get("speed_window_frames", 10),
            stopped_displacement_px=e.get("stopped_displacement_px", 5.0),
            stopped_min_frames=int(e.get("stopped_seconds", 30) * approx_fps),
            stopped_exclude_classes=e.get("stopped_exclude_classes", ["person"]),
        )

        self._stop = False
        signal.signal(signal.SIGINT, self._sigint)
        signal.signal(signal.SIGTERM, self._sigint)

    # ------------------------------------------------------------------ #
    def _sigint(self, *_):
        log.info("Shutdown signal received.")
        self._stop = True

    def _on_stream_down(self, reason: str) -> None:
        ev = self.events_engine.stream_down(reason)
        if ev is not None:
            self.storage.write_events([ev])

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        log.info("Starting main loop. show=%s", self.show)
        last_fps_t = time.time()
        last_fps_n = 0

        try:
            for sf in self.stream.frames():
                if self._stop:
                    break

                # Note we got a frame: clears any STREAM_DOWN state.
                up = self.events_engine.stream_up()
                if up is not None:
                    self.storage.write_events([up])

                detections = self.detector.detect(sf.image)
                tracked, crossings = self.tracker.update(detections)
                feats = self.features.update(
                    tracked, sf.width, sf.height,
                    self.detector.class_name,
                )

                self.storage.write_frame(sf.index, sf.timestamp, feats)
                events = self.events_engine.process(
                    sf.timestamp, feats, crossings,
                )
                if events:
                    self.storage.write_events(events)

                if self.show:
                    annotated = _annotate(
                        sf.image, tracked, self.tracker, feats, self.detector,
                    )
                    cv2.imshow(
                        self.cfg.get("display", {}).get(
                            "window_name", "Livestream Traffic Analytics",
                        ),
                        annotated,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        log.info("Quit key pressed.")
                        break

                # Lightweight FPS log every ~5 s.
                last_fps_n += 1
                now = time.time()
                if now - last_fps_t >= 5.0:
                    fps = last_fps_n / (now - last_fps_t)
                    totals = self.tracker.totals()
                    log.info(
                        "fps=%.1f  vehicles=%d  congestion=%.2f  totals=%s",
                        fps, feats.n_vehicles, feats.congestion, totals,
                    )
                    last_fps_t = now
                    last_fps_n = 0
        finally:
            self.stream.close()
            self.storage.close()
            if self.show:
                cv2.destroyAllWindows()
            log.info("Stopped.")


def cli(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Livestream traffic analytics")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--show", action="store_true",
                   help="Open an OpenCV window with overlays.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Runner(args.config, args.show).run()
    return 0


if __name__ == "__main__":
    sys.exit(cli())

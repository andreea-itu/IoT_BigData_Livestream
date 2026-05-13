"""Per-frame feature extraction.

Computes:
  - n_vehicles (in optional ROI polygon)
  - mean_speed_px (mean displacement of tracked centroids over the
    last `speed_window_frames` frames)
  - congestion: 0..1 normalized = density / (1 + mean_speed_px), squashed.

Also keeps per-track displacement history so we can detect "stopped"
tracks for the event engine.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


@dataclass
class FrameFeatures:
    n_vehicles: int
    mean_speed_px: float
    density: float          # vehicles per 1000 px^2 of (ROI or frame) area
    congestion: float       # 0..1
    per_class: Dict[str, int]
    stopped_track_ids: List[int]
    # class_name (or empty string) per stopped track id, same order
    stopped_classes: List[str]


def _centroids(det: sv.Detections) -> np.ndarray:
    if len(det) == 0:
        return np.empty((0, 2), dtype=float)
    xyxy = det.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


def _point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    # Ray casting (poly: Nx2)
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


class FeatureExtractor:
    def __init__(
        self,
        roi_polygon: Optional[List[List[int]]] = None,
        speed_window_frames: int = 10,
        stopped_displacement_px: float = 5.0,
        stopped_min_frames: int = 30,
        stopped_exclude_classes: Optional[List[str]] = None,
    ) -> None:
        self.roi = (
            np.array(roi_polygon, dtype=float)
            if roi_polygon and len(roi_polygon) >= 3
            else None
        )
        self.window = max(2, int(speed_window_frames))
        self.stopped_displacement_px = float(stopped_displacement_px)
        self.stopped_min_frames = int(stopped_min_frames)
        # Pedestrians naturally wait at lights — excluding them keeps the
        # STOPPED_VEHICLE event semantically correct (incident on roadway).
        self.stopped_exclude_classes = set(
            stopped_exclude_classes
            if stopped_exclude_classes is not None
            else ["person"]
        )
        # tracker_id -> deque of recent (x,y) centroids
        self._history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        # tracker_id -> consecutive frames with low displacement
        self._stopped_run: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    def update(
        self,
        tracked: sv.Detections,
        frame_w: int,
        frame_h: int,
        class_name_fn,
    ) -> FrameFeatures:
        cents = _centroids(tracked)

        # --- ROI filter ---
        roi_mask = np.ones(len(cents), dtype=bool)
        if self.roi is not None and len(cents):
            roi_mask = np.array(
                [_point_in_polygon((c[0], c[1]), self.roi) for c in cents]
            )
        in_roi_idx = np.where(roi_mask)[0]
        n_in = int(roi_mask.sum())

        # --- per-class breakdown ---
        per_class: Dict[str, int] = {}
        if tracked.class_id is not None:
            for i in in_roi_idx:
                cname = class_name_fn(int(tracked.class_id[i]))
                per_class[cname] = per_class.get(cname, 0) + 1

        # --- displacement history + speed proxy ---
        displacements: List[float] = []
        stopped_ids: List[int] = []
        stopped_classes: List[str] = []
        if tracked.tracker_id is not None:
            seen_ids = set()
            for i, tid in enumerate(tracked.tracker_id):
                if tid is None:
                    continue
                tid = int(tid)
                seen_ids.add(tid)
                cname = (
                    class_name_fn(int(tracked.class_id[i]))
                    if tracked.class_id is not None else ""
                )
                hist = self._history[tid]
                cxy = (float(cents[i, 0]), float(cents[i, 1]))
                if hist:
                    px, py = hist[-1]
                    d = float(np.hypot(cxy[0] - px, cxy[1] - py))
                    displacements.append(d)

                    # Pedestrians (and other excluded classes) wait at lights
                    # naturally — don't accumulate "stopped" frames for them.
                    if cname in self.stopped_exclude_classes:
                        self._stopped_run[tid] = 0
                    elif d <= self.stopped_displacement_px:
                        self._stopped_run[tid] += 1
                    else:
                        self._stopped_run[tid] = 0

                    if self._stopped_run[tid] >= self.stopped_min_frames:
                        stopped_ids.append(tid)
                        stopped_classes.append(cname)
                hist.append(cxy)

            # Garbage-collect tracks we no longer see.
            stale = set(self._history) - seen_ids
            for tid in stale:
                self._history.pop(tid, None)
                self._stopped_run.pop(tid, None)

        mean_speed = float(np.mean(displacements)) if displacements else 0.0

        # --- density + congestion ---
        if self.roi is not None:
            # Polygon area via shoelace.
            x = self.roi[:, 0]
            y = self.roi[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        else:
            area = float(frame_w * frame_h)
        area = max(area, 1.0)
        density = (n_in / area) * 1000.0  # vehicles per 1000 px^2

        # Map density+speed -> 0..1 with a soft squash.
        # High density + low speed = high congestion.
        raw = density / (1.0 + mean_speed)
        congestion = float(1.0 - np.exp(-raw / 0.5))  # asymptote at 1
        congestion = float(min(max(congestion, 0.0), 1.0))

        return FrameFeatures(
            n_vehicles=n_in,
            mean_speed_px=mean_speed,
            density=density,
            congestion=congestion,
            per_class=per_class,
            stopped_track_ids=stopped_ids,
            stopped_classes=stopped_classes,
        )

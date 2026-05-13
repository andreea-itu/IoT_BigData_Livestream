"""Offline ML-model benchmark for the live traffic detector.

Pulls N frames from the configured YouTube livestream, runs the TF Hub
detector on each, and prints a summary of *runtime behavior*:

- end-to-end FPS through the detect path
- per-frame inference latency (mean / median / p90 / p95 / max)
- detections per frame (mean / max / total, # frames with zero detections)
- confidence distribution of kept detections
- class distribution

This is not an accuracy benchmark — we have no ground-truth labels for
this camera. See `docs/model.md` for the full discussion.

Run:
    poetry run python scripts/benchmark_model.py
    poetry run python scripts/benchmark_model.py --frames 500 --out data/bench.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Make `src` importable when invoked as `python scripts/benchmark_model.py`
# from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.detector import VehicleDetector  # noqa: E402
from src.stream import LiveStream  # noqa: E402

log = logging.getLogger("benchmark_model")


def _percentile(xs, p: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs), p))


def _format_summary(summary: dict) -> str:
    lat = summary["inference_ms"]
    dpf = summary["detections_per_frame"]
    cd = summary["confidence_distribution"]
    lines = [
        "=" * 64,
        "BENCHMARK SUMMARY",
        "=" * 64,
        f"  model:                {summary['weights']}",
        f"  conf threshold:       {summary['conf_threshold']:.2f}",
        f"  vehicle class ids:    {summary['vehicle_class_ids']}",
        f"  frames benchmarked:   {summary['frames_benchmarked']}",
        f"  wall time:            {summary['wall_seconds']:.2f} s",
        f"  end-to-end FPS:       {summary['frames_per_second_e2e']:.2f}",
        "",
        (
            "  inference latency (ms): "
            f"mean={lat['mean']:.1f} median={lat['median']:.1f} "
            f"p90={lat['p90']:.1f} p95={lat['p95']:.1f} max={lat['max']:.1f}"
        ),
        "",
        (
            "  detections / frame:   "
            f"mean={dpf['mean']:.2f} median={dpf['median']:.1f} "
            f"p90={dpf['p90']:.1f} max={dpf['max']} total={dpf['total']}"
        ),
        (
            "  frames with 0 detections: "
            f"{dpf['frames_with_zero_detections']}"
        ),
        "",
        (
            f"  confidence (filtered >= {summary['conf_threshold']:.2f}): "
            f"mean={cd['mean']:.3f} p50={cd['p50']:.3f} "
            f"p90={cd['p90']:.3f} p99={cd['p99']:.3f} samples={cd['samples']}"
        ),
        "",
        "  class distribution (detection-instances):",
    ]
    if summary["class_distribution"]:
        for cname, n in summary["class_distribution"].items():
            lines.append(f"    {cname:<12} {n:>6}")
    else:
        lines.append("    (no detections kept)")
    lines.append("=" * 64)
    return "\n".join(lines)


def benchmark(
    config_path: str,
    n_frames: int,
    warmup: int,
) -> dict:
    cfg = load_config(config_path)
    s = cfg["stream"]
    m = cfg["model"]

    stream = LiveStream(
        youtube_url=s["url"],
        frame_stride=s.get("frame_stride", 5),
        resize_width=s.get("resize_width", 960),
        reresolve_seconds=s.get("reresolve_seconds", 1800),
        max_read_failures=s.get("max_read_failures", 30),
    )
    detector = VehicleDetector(
        weights=m.get(
            "weights", "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        ),
        conf=m.get("conf", 0.35),
        iou=m.get("iou", 0.5),
        vehicle_class_ids=m.get("vehicle_class_ids"),
        imgsz=m.get("imgsz", 320),
        device=m.get("device", ""),
    )

    frame_iter = stream.frames()

    if warmup > 0:
        log.info("Warming up on %d frames (excluded from stats)…", warmup)
        for _ in range(warmup):
            sf = next(frame_iter)
            detector.detect(sf.image)

    latencies_ms: list[float] = []
    dets_per_frame: list[int] = []
    confidences: list[float] = []
    class_counter: Counter = Counter()

    log.info(
        "Benchmarking %d frames (model=%s, conf>=%.2f)…",
        n_frames, m.get("weights"), detector.conf,
    )
    t_total_0 = time.perf_counter()
    for i in range(n_frames):
        sf = next(frame_iter)
        t0 = time.perf_counter()
        dets = detector.detect(sf.image)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        n = int(len(dets))
        dets_per_frame.append(n)
        if n > 0 and dets.confidence is not None:
            confidences.extend(float(c) for c in dets.confidence)
        if n > 0 and dets.class_id is not None:
            for cid in dets.class_id:
                class_counter[detector.class_name(int(cid))] += 1

        if (i + 1) % 25 == 0 or (i + 1) == n_frames:
            log.info(
                "  %d/%d  last_latency=%.1f ms  n_dets=%d",
                i + 1, n_frames, latencies_ms[-1], n,
            )
    t_total = time.perf_counter() - t_total_0
    stream.close()

    summary = {
        "weights": m.get("weights"),
        "conf_threshold": float(detector.conf),
        "vehicle_class_ids": sorted(int(c) for c in detector.vehicle_class_ids),
        "frames_benchmarked": len(latencies_ms),
        "warmup_frames": int(warmup),
        "wall_seconds": float(t_total),
        "frames_per_second_e2e": (
            len(latencies_ms) / t_total if t_total > 0 else 0.0
        ),
        "inference_ms": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "median": float(np.median(latencies_ms)) if latencies_ms else 0.0,
            "p90": _percentile(latencies_ms, 90),
            "p95": _percentile(latencies_ms, 95),
            "max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        },
        "detections_per_frame": {
            "mean": float(np.mean(dets_per_frame)) if dets_per_frame else 0.0,
            "median": (
                float(np.median(dets_per_frame)) if dets_per_frame else 0.0
            ),
            "p90": _percentile(dets_per_frame, 90),
            "max": int(np.max(dets_per_frame)) if dets_per_frame else 0,
            "total": int(np.sum(dets_per_frame)) if dets_per_frame else 0,
            "frames_with_zero_detections": int(
                sum(1 for n in dets_per_frame if n == 0)
            ),
        },
        "confidence_distribution": {
            "mean": float(np.mean(confidences)) if confidences else 0.0,
            "p50": _percentile(confidences, 50),
            "p90": _percentile(confidences, 90),
            "p99": _percentile(confidences, 99),
            "min": float(np.min(confidences)) if confidences else 0.0,
            "max": float(np.max(confidences)) if confidences else 0.0,
            "samples": len(confidences),
        },
        "class_distribution": dict(class_counter.most_common()),
    }
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark the TF Hub detector against the configured "
            "YouTube live stream and print runtime behavior stats."
        )
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    p.add_argument(
        "--frames", type=int, default=200,
        help="Number of frames to benchmark (default: 200)",
    )
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup frames to discard before measuring (default: 5)",
    )
    p.add_argument(
        "--out", default=None,
        help="Optional path to write the summary as JSON.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = benchmark(
        config_path=args.config,
        n_frames=int(args.frames),
        warmup=int(args.warmup),
    )

    print()
    print(_format_summary(summary))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


#poetry run python scripts/benchmark_model.py --frames 200 --warmup 5
#poetry run python scripts/benchmark_model.py --frames 500 --out data/model_bench.json
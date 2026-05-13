"""TensorFlow / TF Hub object detector.

Wraps a COCO-trained SSD MobileNet V2 SavedModel (or any compatible
`tf.saved_model` from TF Hub: SSD MobileNet, EfficientDet, CenterNet,
…) so the rest of the pipeline (tracker, line zones, annotators, event
engine) can keep consuming the standard `supervision.Detections` data
model with no other changes.

Why TF Hub and not a custom model?
- It ships pre-trained on COCO (80 classes including person, car,
  bicycle, motorcycle, bus, truck) — exactly what we need for street
  livestreams, with no training step.
- It's `tf.saved_model.load`-able from a URL, so the install procedure
  stays a one-liner.

Class IDs follow the **TF Object Detection API COCO label map**, which
is **1-indexed**:

    1 person | 2 bicycle | 3 car | 4 motorcycle | 6 bus | 8 truck

(That is different from the 0-indexed Ultralytics convention. The
config file `vehicle_class_ids` is in TF convention.)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

# Quiet TF logs unless something is actually broken.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import supervision as sv
import tensorflow as tf
import tensorflow_hub as hub

log = logging.getLogger(__name__)


# TF Object Detection API COCO label map (1-indexed).
COCO_VEHICLES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class VehicleDetector:
    """COCO object detector backed by a TF Hub SavedModel."""

    def __init__(
        self,
        weights: str = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        conf: float = 0.35,
        iou: float = 0.5,                 # accepted for parity, ignored by SSD
        vehicle_class_ids: Optional[List[int]] = None,
        imgsz: int = 320,                 # accepted for parity (SSD is fixed-shape)
        device: str = "",
    ) -> None:
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.vehicle_class_ids = set(
            vehicle_class_ids if vehicle_class_ids is not None
            else COCO_VEHICLES.keys()
        )

        self._configure_device(device)

        log.info("Loading TF Hub model %s (this may download on first run)…",
                 weights)
        loaded = hub.load(weights)
        # TF Hub object-detection SavedModels expose a default callable.
        self._infer = loaded.signatures["serving_default"] \
            if hasattr(loaded, "signatures") and "serving_default" in loaded.signatures \
            else loaded
        log.info("TF detector ready.")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _configure_device(device: str) -> None:
        device = (device or "").lower()
        if device in ("", "cpu"):
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:  # noqa: BLE001
                pass
        # If user wants GPU we just leave TF defaults — TF will pick CUDA if
        # available. (No need to do anything special for "cuda:0" here.)

    # ------------------------------------------------------------------ #
    def class_name(self, class_id: int) -> str:
        return COCO_VEHICLES.get(int(class_id), f"class_{int(class_id)}")

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Run detection on one BGR frame and return filtered detections."""
        # TF Hub detection models expect uint8 RGB tensors of shape
        # [1, H, W, 3]. They handle resize internally.
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB without an extra cv2 call
        tensor = tf.convert_to_tensor(rgb[np.newaxis, ...], dtype=tf.uint8)

        out = self._infer(tensor)

        # Output keys for object-detection models from TF Hub.
        boxes = out["detection_boxes"].numpy()[0]      # [N,4] ymin,xmin,ymax,xmax in [0,1]
        classes = out["detection_classes"].numpy()[0].astype(int)
        scores = out["detection_scores"].numpy()[0]

        # Filter by score and target classes.
        mask = (scores >= self.conf) & np.isin(
            classes, list(self.vehicle_class_ids)
        )
        if not mask.any():
            return sv.Detections.empty()

        boxes = boxes[mask]
        classes = classes[mask]
        scores = scores[mask]

        # Convert normalized [ymin,xmin,ymax,xmax] -> pixel [x1,y1,x2,y2].
        h, w = frame_bgr.shape[:2]
        ymin, xmin, ymax, xmax = boxes.T
        xyxy = np.stack(
            [xmin * w, ymin * h, xmax * w, ymax * h], axis=1,
        ).astype(np.float32)

        return sv.Detections(
            xyxy=xyxy,
            confidence=scores.astype(np.float32),
            class_id=classes.astype(int),
        )

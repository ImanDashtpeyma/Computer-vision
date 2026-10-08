"""Glue between a camera frame, an inference backend, and a drawable result."""
from __future__ import annotations

import numpy as np

from .backends import Detection, InferenceBackend


class ObjectDetector:
    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def run_on_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections = self.backend.detect(frame)
        annotated = self.annotate(frame, detections)
        return annotated, detections

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        import cv2

        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{det.label} {det.confidence:.0%}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,
            )
        return annotated

"""Thin wrapper around OpenCV's VideoCapture for the Orange Pi camera.

Kept deliberately small: the only job here is "give me frames", so the
detection code and the tests don't need to know anything about V4L2 or
OpenCV capture quirks.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .config import CameraConfig


class Camera:
    """Context-manager wrapper around cv2.VideoCapture.

    Example:
        with Camera(config.camera) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap = None

    def open(self) -> None:
        import cv2  # imported lazily so importing this module never requires

        if self.config.use_v4l2:
            self._cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(self.config.device)
        if self.config.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device {self.config.device!r}. "
                "On the Orange Pi, load the camera driver first "
                "(`sudo modprobe vin_v4l2`), then check which /dev/video* node is the "
                "OV13850 (`ls /dev/video*`; the manual lists /dev/video0 and /dev/video8). "
                "`python scripts/check_camera.py` tries the common options for you."
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera is not open; call open() or use as a context manager.")
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

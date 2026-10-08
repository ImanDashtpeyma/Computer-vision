"""Wraps MediaPipe's Pose Landmarker (Tasks API) behind a small, testable
interface, so route code never has to know MediaPipe's API shape -- it
just asks for landmarks.

This module is intentionally generic: it detects body pose, full stop. The
thesis prototype ("Feeling the Correction") builds feedback logic on top
of this, but that logic -- and any participant data -- stays out of this
public repo (see README).
"""
from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# BlazePose's 33-point topology (index pairs into PoseResult.landmarks),
# used only for drawing a skeleton overlay in the /pose/annotate endpoint.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass
class PoseResult:
    pose_detected: bool
    landmarks: list[Landmark]


class PoseEstimator:
    """Thin, thread-safe wrapper around mediapipe's PoseLandmarker.

    The model is loaded lazily on first `detect()` call, not at import
    time or construction, so importing/instantiating this class never
    requires the model file or system GL libraries to be present (useful
    for tests -- see tests/test_pose_endpoint.py, which never exercises
    this class directly and uses a fake instead).
    """

    def __init__(self, model_path: pathlib.Path = MODEL_PATH):
        self._model_path = model_path
        self._landmarker = None
        self._mp = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._landmarker is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {self._model_path}. "
                "Run `python scripts/download_model.py` first."
            )

        import mediapipe as mp

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self._model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, frame_rgb: np.ndarray) -> PoseResult:
        self._ensure_loaded()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

        with self._lock:
            result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseResult(pose_detected=False, landmarks=[])

        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility, presence=lm.presence)
            for lm in result.pose_landmarks[0]
        ]
        return PoseResult(pose_detected=True, landmarks=landmarks)


_estimator: Optional[PoseEstimator] = None


def get_pose_estimator() -> PoseEstimator:
    """FastAPI dependency: a module-level singleton so the model is loaded
    (at most) once per process, not once per request."""
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator()
    return _estimator

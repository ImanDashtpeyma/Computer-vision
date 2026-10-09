"""Tests for the pose endpoints that run on any machine with no GPU/EGL
libraries and no downloaded model -- the real PoseEstimator is swapped
for a fake via FastAPI's dependency override. A real-model, real-mediapipe
check lives in scripts/smoke_test.py (manual, not part of CI -- see its
docstring for why).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pose_estimation import Landmark, PoseResult, get_pose_estimator


class FakePoseEstimator:
    def __init__(self, result: PoseResult):
        self._result = result

    def detect(self, frame_rgb: np.ndarray) -> PoseResult:
        return self._result


def make_png_bytes(width: int = 32, height: int = 32) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return encoded.tobytes()


def png_upload() -> dict:
    return {"file": ("frame.png", make_png_bytes(), "image/png")}


def override_with(result: PoseResult) -> TestClient:
    app.dependency_overrides[get_pose_estimator] = lambda: FakePoseEstimator(result)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_detect_pose_returns_landmarks():
    pose_detected_result = PoseResult(
        pose_detected=True,
        landmarks=[Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9, presence=0.9) for _ in range(33)],
    )
    client = override_with(pose_detected_result)

    response = client.post("/pose/detect", files=png_upload())

    assert response.status_code == 200
    body = response.json()
    assert body["pose_detected"] is True
    assert len(body["landmarks"]) == 33
    assert body["landmarks"][0]["x"] == 0.5


def test_detect_pose_with_no_pose_found():
    client = override_with(PoseResult(pose_detected=False, landmarks=[]))

    response = client.post("/pose/detect", files=png_upload())

    assert response.status_code == 200
    assert response.json() == {"pose_detected": False, "landmarks": []}


def test_detect_pose_rejects_undecodable_file():
    client = override_with(PoseResult(pose_detected=False, landmarks=[]))

    response = client.post(
        "/pose/detect", files={"file": ("not-an-image.txt", b"hello world", "text/plain")}
    )

    assert response.status_code == 400


def test_annotate_pose_returns_png_with_skeleton():
    pose_detected_result = PoseResult(
        pose_detected=True,
        landmarks=[Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9, presence=0.9) for _ in range(33)],
    )
    client = override_with(pose_detected_result)

    response = client.post("/pose/annotate", files=png_upload())

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_annotate_pose_without_detection_returns_original_sized_image():
    client = override_with(PoseResult(pose_detected=False, landmarks=[]))

    response = client.post("/pose/annotate", files=png_upload())

    assert response.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (32, 32)

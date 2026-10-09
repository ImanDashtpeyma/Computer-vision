"""Tests for the detection pipeline that run on any machine, with no
camera and no real model weights -- this is what CI runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backends import Detection, InferenceBackend, NPUBackend, build_backend
from src.config import ModelConfig
from src.detector import ObjectDetector


class FakeBackend(InferenceBackend):
    """A stand-in backend so detector/annotation logic can be tested
    without loading real model weights."""

    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detections


def make_frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_object_detector_returns_same_detections_backend_gives():
    expected = [Detection(label="person", confidence=0.91, box=(5, 5, 20, 20))]
    detector = ObjectDetector(FakeBackend(expected))

    annotated, detections = detector.run_on_frame(make_frame())

    assert detections == expected
    assert annotated.shape == make_frame().shape


def test_annotate_does_not_mutate_input_frame():
    frame = make_frame()
    original = frame.copy()
    detections = [Detection(label="car", confidence=0.8, box=(1, 1, 10, 10))]

    ObjectDetector.annotate(frame, detections)

    assert np.array_equal(frame, original)


def test_annotate_with_no_detections_returns_unchanged_image():
    frame = make_frame()
    annotated = ObjectDetector.annotate(frame, [])
    assert np.array_equal(annotated, frame)


def test_build_backend_rejects_unknown_backend_name():
    with pytest.raises(ValueError):
        build_backend(ModelConfig(backend="quantum"))


def test_build_backend_returns_npu_backend_without_touching_hardware():
    # Constructing the NPU backend must not load the driver, library or .nb
    # file -- only the first detect() call does.
    backend = build_backend(ModelConfig(backend="npu"))
    assert isinstance(backend, NPUBackend)

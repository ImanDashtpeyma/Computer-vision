"""Inference backends.

The pipeline is split from the inference engine on purpose: `Detection`
objects are a plain, backend-agnostic format, and swapping how they get
produced (CPU today, NPU later) should never touch camera.py or main.py.

Backend status, honestly:

* OpenCVDNNBackend ("cpu") is fully implemented, works on any machine with
  OpenCV, and is what this repo actually runs and tests in CI.

* NPUBackend ("npu") is a documented stub, not a working implementation.
  The Allwinner A733's VeriSilicon VIP9000 NPU *does* work on this specific
  board -- I've run it successfully through Allwinner/Orange Pi's official
  demo tooling -- but as of writing there is no mainstream, scriptable path
  into it: ONNX Runtime only exposes CPU on this SoC (a VIPLite execution
  provider is an open proposal, not shipped code -- see
  https://github.com/microsoft/onnxruntime/issues/28244), and the real
  route is Allwinner's ACUITY Toolkit converting a model to its NBG format
  and calling it through the low-level `libVIPhal.so` (VIPLite) API
  directly. That toolchain is vendor-specific, tied to whatever SDK/image
  shipped with the board, and not something that can be wired up and
  verified from outside the device. Rather than fake that integration,
  this class documents the real steps and raises clearly until someone
  (me, with the board in front of me) fills it in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates


class InferenceBackend(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR frame."""
        raise NotImplementedError


class OpenCVDNNBackend(InferenceBackend):
    """CPU inference using OpenCV's DNN module and a pretrained
    MobileNet-SSD (VOC0712 weights, trained by chuanqi305, not by me --
    see README for provenance and license).
    """

    def __init__(self, config: ModelConfig):
        import cv2  # lazy import, see Camera for rationale

        self.config = config
        self.net = cv2.dnn.readNetFromCaffe(config.prototxt_path, config.weights_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2

        h, w = frame.shape[:2]
        size = self.config.input_size
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (size, size)), scalefactor=0.007843,
            size=(size, size), mean=127.5,
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        detections: list[Detection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self.config.confidence_threshold:
                continue
            class_id = int(raw[0, 0, i, 1])
            if class_id < 0 or class_id >= len(self.config.classes):
                continue
            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                Detection(
                    label=self.config.classes[class_id],
                    confidence=confidence,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections


class NPUBackend(InferenceBackend):
    """Documented stub for Allwinner A733 NPU (VIP9000) inference.

    Not implemented yet -- see the module docstring above for exactly why,
    and the README's "NPU status" section for the steps this would need:
    1. Convert the model to Allwinner's NBG format with the ACUITY Toolkit.
    2. Load/run it through libVIPhal.so (the VIPLite API) instead of
       OpenCV/ONNX Runtime.
    3. Wrap that in this class so `detect()` returns the same `Detection`
       objects the CPU backend returns -- the rest of the app doesn't change.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError(
            "NPU backend is not wired up in this repo yet. The A733's "
            "VIP9000 NPU works on this board via Allwinner/Orange Pi's own "
            "demo tooling, but integrating it here needs the vendor ACUITY "
            "Toolkit (model -> NBG) and the libVIPhal.so API, which is "
            "board/image-specific and documented in the README instead of "
            "faked here. Use backend='cpu' for a working, portable path."
        )


def build_backend(config: ModelConfig) -> InferenceBackend:
    if config.backend == "cpu":
        return OpenCVDNNBackend(config)
    if config.backend == "npu":
        return NPUBackend(config)
    raise ValueError(f"Unknown backend {config.backend!r}; expected 'cpu' or 'npu'.")

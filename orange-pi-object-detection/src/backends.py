"""Inference backends.

The pipeline is split from the inference engine on purpose: `Detection`
objects are a plain, backend-agnostic format, and swapping how they get
produced (CPU or NPU) never touches camera.py or main.py.

* OpenCVDNNBackend ("cpu"): MobileNet-SSD through OpenCV's DNN module. Works
  on any machine with OpenCV and is what CI exercises.

* NPUBackend ("npu"): YOLOv5s on the Allwinner A733's NPU, through the vendor
  `awnn` helper library (built from Allwinner's ai-sdk by
  native/build_awnn.sh) and a prebuilt `.nb` network binary. Only the network
  runs on the NPU; letterbox pre-processing and box decoding/NMS are numpy
  (src/yolov5.py). This path needs the real board, the vendor driver and the
  `.nb` file, so it is not exercised in CI -- its maths is unit-tested with
  synthetic tensors, and the hardware path is checked by hand on the board.
"""
from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod

import numpy as np

from .backends_types import Detection
from .config import ModelConfig

__all__ = ["Detection", "InferenceBackend", "OpenCVDNNBackend", "NPUBackend", "build_backend"]


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
    """YOLOv5s on the Allwinner A733 NPU (VIP9000, NPU v3) via the awnn library.

    Everything hardware-related is loaded lazily on the first `detect()` call,
    so constructing this class (e.g. in tests, or on a machine without the
    NPU) never touches the driver. A fake `lib` can be injected for testing.
    """

    # yolov5s outputs: 3 anchors x grid x grid x 85 floats for strides 8/16/32
    OUTPUT_SIZES = [3 * g * g * 85 for g in (80, 40, 20)]

    def __init__(self, config: ModelConfig, lib=None):
        self.config = config
        self._lib = lib
        self._ctx = None

    def _ensure_ready(self) -> None:
        if self._ctx is not None:
            return
        nbg = pathlib.Path(self.config.nbg_path)
        if not nbg.exists():
            raise FileNotFoundError(
                f"NPU network binary not found at {nbg}. Copy yolov5.nb (the v3 build) from "
                "Allwinner's ai-sdk examples/yolov5/model/v3/ -- see README, 'NPU backend'."
            )
        if self._lib is None:
            from .awnn import AwnnLibrary

            self._lib = AwnnLibrary(self.config.awnn_lib_path)
        self._lib.init()
        self._ctx = self._lib.create(str(nbg))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from .yolov5 import decode_outputs, letterbox_preprocess

        self._ensure_ready()
        size = self.config.npu_input_size
        chw, info = letterbox_preprocess(frame, size)
        outputs = self._lib.run(self._ctx, chw, self.OUTPUT_SIZES)
        return decode_outputs(
            outputs,
            info,
            self.config.npu_classes,
            size=size,
            conf_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.npu_nms_threshold,
        )

    def close(self) -> None:
        if self._ctx is not None:
            self._lib.destroy(self._ctx)
            self._lib.uninit()
            self._ctx = None


def build_backend(config: ModelConfig) -> InferenceBackend:
    if config.backend == "cpu":
        return OpenCVDNNBackend(config)
    if config.backend == "npu":
        return NPUBackend(config)
    raise ValueError(f"Unknown backend {config.backend!r}; expected 'cpu' or 'npu'.")

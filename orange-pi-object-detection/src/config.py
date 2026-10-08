"""Runtime configuration for the object detection pipeline.

Everything that might need to change between a dev laptop and the actual
Orange Pi (camera device path, resolution, which inference backend to use)
lives here so it is not hard-coded inside the detection loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Labels for the pretrained MobileNet-SSD (VOC0712) model used by the CPU
# backend. Index position matches the class id the network outputs.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


@dataclass
class CameraConfig:
    # On the Orange Pi this is normally the V4L2 node exposed for the
    # OV13850 MIPI camera (e.g. /dev/video0). On a dev machine with a USB
    # webcam this is also usually /dev/video0 or an integer index.
    device: str | int = 0
    capture_width: int = 1280
    capture_height: int = 720
    # The 13MP sensor can go much higher; we deliberately capture at a
    # lower resolution because the model and the CPU, not the sensor, are
    # the bottleneck for real-time detection.


@dataclass
class ModelConfig:
    backend: str = "cpu"  # "cpu" (default, works everywhere) or "npu" (Orange Pi only, see README)
    prototxt_path: str = "models/deploy.prototxt"
    weights_path: str = "models/mobilenet_iter_73000.caffemodel"
    input_size: int = 300  # MobileNet-SSD expects 300x300 input
    confidence_threshold: float = 0.5
    classes: list[str] = field(default_factory=lambda: list(VOC_CLASSES))


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    show_window: bool = True
    save_output_path: str | None = None  # e.g. "output/demo.mp4", None to disable

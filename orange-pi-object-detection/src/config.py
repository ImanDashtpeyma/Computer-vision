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

# The 80 COCO labels the YOLOv5s .nb model from Allwinner's ai-sdk was trained
# on. Order matches the class index the network outputs.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class CameraConfig:
    # On the Orange Pi this is normally the V4L2 node exposed for the
    # OV13850 MIPI camera (e.g. /dev/video0). On a dev machine with a USB
    # webcam this is also usually /dev/video0 or an integer index.
    device: str | int = 0
    capture_width: int = 1280
    capture_height: int = 720
    # Force the V4L2 backend and (optionally) a pixel format such as "MJPG" or
    # "YUYV". The Orange Pi's MIPI camera goes through the sunxi vin_v4l2
    # driver, so which of these works has to be checked on the board -- see
    # scripts/check_camera.py.
    use_v4l2: bool = True
    fourcc: str | None = None
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

    # --- NPU backend (Allwinner A733 / VIPLite), only used when backend == "npu" ---
    nbg_path: str = "models/yolov5.nb"  # network binary (.nb) for NPU v3
    awnn_lib_path: str = "native/libawnn_npu.so"  # built by native/build_awnn.sh
    npu_input_size: int = 640  # the yolov5 .nb takes a fixed 640x640 letterboxed RGB image
    npu_nms_threshold: float = 0.45
    npu_classes: list[str] = field(default_factory=lambda: list(COCO_CLASSES))


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    show_window: bool = True
    save_output_path: str | None = None  # e.g. "output/demo.mp4", None to disable

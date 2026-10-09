"""YOLOv5 pre- and post-processing for the Allwinner ai-sdk `yolov5.nb` model.

This is a numpy port of the processing in the vendor's C++ demo
(examples/yolov5/yolov5_pre_process.cpp and yolov5_post_process.cpp in
Allwinner's ai-sdk), so it can be unit-tested on any machine without the NPU.
The NPU itself only runs the network; everything before and after is plain
image/array maths.

Contract with the .nb model (A733, NPU v3):
* input : uint8 RGB, planar CHW, 640x640, letterboxed with black padding.
          Mean/scale (1/255) are baked into the network, so raw 0-255 pixel
          values go in.
* output: three float32 tensors of raw logits, laid out [anchor][h][w][85]
          for strides 8 (80x80), 16 (40x40) and 32 (20x20), in that order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends_types import Detection

# YOLOv5 default anchors (w, h) per output stride.
ANCHORS = {
    8: [(10, 13), (16, 30), (33, 23)],
    16: [(30, 61), (62, 45), (59, 119)],
    32: [(116, 90), (156, 198), (373, 326)],
}
STRIDES = (8, 16, 32)  # order of the NPU outputs
NUM_ATTRS = 85  # x, y, w, h, objectness + 80 class scores


@dataclass
class LetterboxInfo:
    """How the original frame was placed inside the square network input."""

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


def letterbox_preprocess(
    frame_bgr: np.ndarray, size: int = 640
) -> tuple[np.ndarray, LetterboxInfo]:
    """BGR frame -> contiguous uint8 RGB CHW array of shape (3, size, size)."""
    import cv2

    h, w = frame_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(scale * w), int(scale * h)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h))

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    return chw, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_w=w, orig_h=h)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy, class-agnostic NMS (same behaviour as the vendor demo).

    boxes are (x1, y1, x2, y2). Returns indices of kept boxes, best first.
    """
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_outputs(
    outputs: list[np.ndarray],
    info: LetterboxInfo,
    classes: list[str],
    size: int = 640,
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    """Turn the three raw NPU output tensors into Detection objects in the
    coordinates of the original (un-letterboxed) frame."""
    if len(outputs) != len(STRIDES):
        raise ValueError(f"Expected {len(STRIDES)} output tensors, got {len(outputs)}.")

    all_boxes, all_scores, all_labels = [], [], []
    for stride, flat in zip(STRIDES, outputs):
        grid = size // stride
        expected = 3 * grid * grid * NUM_ATTRS
        if flat.size != expected:
            raise ValueError(
                f"Output for stride {stride} has {flat.size} values, expected {expected}."
            )
        feat = flat.reshape(3, grid, grid, NUM_ATTRS)

        obj = _sigmoid(feat[..., 4])
        cls_logits = feat[..., 5:]
        cls_idx = cls_logits.argmax(axis=-1)
        cls_score = _sigmoid(np.take_along_axis(cls_logits, cls_idx[..., None], axis=-1)[..., 0])
        score = obj * cls_score

        a_idx, h_idx, w_idx = np.nonzero(score >= conf_threshold)
        if a_idx.size == 0:
            continue

        raw = feat[a_idx, h_idx, w_idx]
        dx, dy, dw, dh = (_sigmoid(raw[:, i]) for i in range(4))
        anchors = np.array(ANCHORS[stride], dtype=np.float32)[a_idx]
        cx = (dx * 2.0 - 0.5 + w_idx) * stride
        cy = (dy * 2.0 - 0.5 + h_idx) * stride
        bw = dw * dw * 4.0 * anchors[:, 0]
        bh = dh * dh * 4.0 * anchors[:, 1]

        all_boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
        all_scores.append(score[a_idx, h_idx, w_idx])
        all_labels.append(cls_idx[a_idx, h_idx, w_idx])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    detections: list[Detection] = []
    for i in _nms(boxes, scores, nms_threshold):
        x1, y1, x2, y2 = boxes[i]
        x1 = np.clip((x1 - info.pad_x) / info.scale, 0, info.orig_w - 1)
        x2 = np.clip((x2 - info.pad_x) / info.scale, 0, info.orig_w - 1)
        y1 = np.clip((y1 - info.pad_y) / info.scale, 0, info.orig_h - 1)
        y2 = np.clip((y2 - info.pad_y) / info.scale, 0, info.orig_h - 1)
        detections.append(
            Detection(
                label=classes[int(labels[i])],
                confidence=float(scores[i]),
                box=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return detections

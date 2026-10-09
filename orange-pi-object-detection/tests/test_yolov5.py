"""YOLOv5 pre/post-processing tests with synthetic tensors (no NPU needed)."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import COCO_CLASSES
from src.yolov5 import NUM_ATTRS, decode_outputs, letterbox_preprocess

SIZE = 640
OUTPUT_SIZES = [3 * g * g * NUM_ATTRS for g in (80, 40, 20)]


def empty_outputs() -> list[np.ndarray]:
    # Logits of 0 -> sigmoid 0.5 -> score 0.25, below the 0.4 threshold.
    return [np.zeros(n, dtype=np.float32) for n in OUTPUT_SIZES]


def set_cell(outputs, stride_index, anchor, h, w, cls, obj=8.0, cls_logit=8.0):
    grid = (80, 40, 20)[stride_index]
    feat = outputs[stride_index].reshape(3, grid, grid, NUM_ATTRS)
    feat[anchor, h, w, 4] = obj
    feat[anchor, h, w, 5 + cls] = cls_logit


def square_info():
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    return letterbox_preprocess(frame, SIZE)[1]


def test_letterbox_output_is_uint8_rgb_chw():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR

    chw, info = letterbox_preprocess(frame, SIZE)

    assert chw.shape == (3, SIZE, SIZE)
    assert chw.dtype == np.uint8
    assert chw.flags["C_CONTIGUOUS"]
    # BGR (10,20,30) -> RGB channel order (30,20,10) in the picture area
    assert tuple(chw[:, SIZE // 2, SIZE // 2]) == (30, 20, 10)


def test_letterbox_pads_wide_frames_top_and_bottom_with_black():
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)

    chw, info = letterbox_preprocess(frame, SIZE)

    assert info.scale == 1.0
    assert (info.pad_x, info.pad_y) == (0, 140)
    assert chw[:, 0:140, :].max() == 0
    assert chw[:, 500:, :].max() == 0


def test_no_detections_when_everything_is_below_threshold():
    assert decode_outputs(empty_outputs(), square_info(), COCO_CLASSES) == []


def test_single_cell_decodes_to_expected_box_and_label():
    outputs = empty_outputs()
    # stride 32, anchor 1 -> (156, 198), cell (h=10, w=5), class 0 = person
    set_cell(outputs, 2, anchor=1, h=10, w=5, cls=0)

    detections = decode_outputs(outputs, square_info(), COCO_CLASSES)

    assert len(detections) == 1
    det = detections[0]
    assert det.label == "person"
    assert det.confidence > 0.99
    # centre = ((0.5*2-0.5+5)*32, (0.5*2-0.5+10)*32) = (176, 336); size = 156 x 198
    assert det.box == (98, 237, 254, 435)


def test_boxes_are_mapped_back_through_the_letterbox():
    outputs = empty_outputs()
    set_cell(outputs, 2, anchor=1, h=10, w=5, cls=2)  # car
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # scale 0.5, pad_y 140
    info = letterbox_preprocess(frame, SIZE)[1]

    det = decode_outputs(outputs, info, COCO_CLASSES)[0]

    assert det.label == "car"
    # x: 98/0.5 .. 254/0.5 ; y: (237-140)/0.5 .. (435-140)/0.5
    assert det.box == (196, 194, 508, 590)


def test_overlapping_duplicates_are_suppressed_by_nms():
    outputs = empty_outputs()
    set_cell(outputs, 2, anchor=1, h=10, w=5, cls=0, obj=8.0, cls_logit=8.0)
    # neighbouring cell, lower score
    set_cell(outputs, 2, anchor=1, h=10, w=6, cls=0, obj=6.0, cls_logit=6.0)

    detections = decode_outputs(outputs, square_info(), COCO_CLASSES, nms_threshold=0.3)

    assert len(detections) == 1


def test_wrong_tensor_count_or_size_is_rejected():
    with pytest.raises(ValueError):
        decode_outputs(empty_outputs()[:2], square_info(), COCO_CLASSES)
    bad = empty_outputs()
    bad[0] = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        decode_outputs(bad, square_info(), COCO_CLASSES)

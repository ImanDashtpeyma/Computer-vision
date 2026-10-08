#!/usr/bin/env python3
"""Manual end-to-end check: real model, real MediaPipe inference.

Not run as part of pytest/CI on purpose -- it needs the model downloaded
and system GL libraries installed (libegl1/libgl1, see README), neither of
which CI should be required to provide just to run the unit tests. Run
this yourself after `pip install -r requirements.txt` and
`python scripts/download_model.py`:

    python scripts/smoke_test.py path/to/a/photo/of/a/person.jpg
"""
from __future__ import annotations

import sys

import cv2

from app.pose_estimation import PoseEstimator


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    image_path = sys.argv[1]
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Could not read image: {image_path}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    estimator = PoseEstimator()
    result = estimator.detect(rgb)

    print(f"pose_detected: {result.pose_detected}")
    print(f"landmark count: {len(result.landmarks)}")
    if result.landmarks:
        print("first landmark:", result.landmarks[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

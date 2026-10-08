#!/usr/bin/env python3
"""Fetch the MediaPipe Pose Landmarker model bundle.

Not committed to git. "lite" is the fastest variant and is plenty for a
demo API; swap VARIANT for "full" or "heavy" for more accurate (slower)
landmarks -- see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request

VARIANT = "lite"
URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{VARIANT}/float16/1/pose_landmarker_{VARIANT}.task"
)


def main() -> int:
    models_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / f"pose_landmarker_{VARIANT}.task"

    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (already present): {dest}")
        return 0

    print(f"downloading {URL} -> {dest}")
    urllib.request.urlretrieve(URL, dest)
    print(f"  ok, {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

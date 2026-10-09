#!/usr/bin/env python3
"""Try the usual ways of opening the Orange Pi's MIPI camera and report what works.

Run on the board after `sudo modprobe vin_v4l2`:

    python scripts/check_camera.py            # tries /dev/video0 and /dev/video8
    python scripts/check_camera.py /dev/video8
"""
from __future__ import annotations

import glob
import sys

import cv2


def try_open(device: str, fourcc: str | None) -> str:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        if not cap.isOpened():
            return "could not open"
        ok, frame = cap.read()
        if not ok or frame is None:
            return "opened, but read() returned no frame"
        return f"OK, frame {frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def main() -> int:
    devices = sys.argv[1:] or sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* nodes. Did you run `sudo modprobe vin_v4l2`?")
        return 1
    for device in devices:
        for fourcc in (None, "MJPG", "YUYV"):
            print(f"{device:14s} fourcc={fourcc or 'default':8s} -> {try_open(device, fourcc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

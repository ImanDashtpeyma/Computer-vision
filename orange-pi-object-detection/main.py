#!/usr/bin/env python3
"""CLI entrypoint: run live object detection from the Orange Pi camera.

    python main.py --backend cpu --no-window --save output/demo.mp4

Run `python scripts/download_model.py` first to fetch the model weights.
"""
from __future__ import annotations

import argparse

from src.backends import build_backend
from src.camera import Camera
from src.config import AppConfig
from src.detector import ObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=0, help="Camera device path or index (default: 0)")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "npu"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-window", action="store_true", help="Don't open a display window")
    parser.add_argument("--save", default=None, help="Optional path to save annotated output video")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig()
    config.camera.device = args.device
    config.model.backend = args.backend
    config.model.confidence_threshold = args.confidence
    config.show_window = not args.no_window
    config.save_output_path = args.save

    backend = build_backend(config.model)
    detector = ObjectDetector(backend)

    import cv2

    writer = None
    with Camera(config.camera) as cam:
        for frame in cam.frames():
            annotated, detections = detector.run_on_frame(frame)

            if detections:
                summary = ", ".join(f"{d.label} ({d.confidence:.0%})" for d in detections)
                print(summary)

            if config.save_output_path:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(config.save_output_path, fourcc, 20.0, (w, h))
                writer.write(annotated)

            if config.show_window:
                cv2.imshow("orange-pi-object-detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if writer is not None:
        writer.release()
    if config.show_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

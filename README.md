# Computer Vision

Two small computer-vision projects that run on real hardware or serve real requests, each with tests, Docker and CI. They live side by side in this repo.

| Project | What it is | Stack |
|---|---|---|
| [`orange-pi-object-detection`](orange-pi-object-detection) | Live object detection from a 13MP MIPI camera on an Orange Pi 4 Pro, with a portable CPU backend and an NPU backend for the board's Allwinner A733 | Python, OpenCV, numpy, ctypes, a bit of C |
| [`fastapi-pose-detection`](fastapi-pose-detection) | HTTP API that finds 33 body-pose keypoints in an uploaded image and can return the image with a skeleton drawn on it | Python, FastAPI, MediaPipe Pose Landmarker |

[![Orange Pi CI](https://github.com/ImanDashtpeyma/Computer-vision/actions/workflows/orange-pi-ci.yml/badge.svg)](https://github.com/ImanDashtpeyma/Computer-vision/actions/workflows/orange-pi-ci.yml)
[![FastAPI CI](https://github.com/ImanDashtpeyma/Computer-vision/actions/workflows/fastapi-pose-ci.yml/badge.svg)](https://github.com/ImanDashtpeyma/Computer-vision/actions/workflows/fastapi-pose-ci.yml)

## A note on what's mine

None of the models here were trained by me. They are public, pretrained networks (MobileNet-SSD, YOLOv5s from Allwinner's SDK, and Google's MediaPipe pose model). The work in this repo is everything around them: capturing frames, getting a model to run on the NPU of a very new chip, pre- and post-processing, the API, and the tests/Docker/CI that keep it honest.

## Orange Pi object detection

The Orange Pi 4 Pro uses the Allwinner A733, whose 3 TOPS NPU has little public tooling so far (ONNX Runtime only sees the CPU on it). The project handles that with two interchangeable backends behind one small interface:

- **`cpu`** runs MobileNet-SSD through OpenCV's DNN module. It works on any Linux machine with a camera, and it's what CI exercises.
- **`npu`** runs YOLOv5s on the NPU through Allwinner's `awnn` helper library, loaded from Python with `ctypes`. The letterbox and box-decoding steps are a numpy port of the vendor's C++ demo, so they can be unit-tested without the board.

Details, setup and the on-board steps are in the [project README](orange-pi-object-detection/README.md).

## FastAPI pose detection

`POST /pose/detect` returns the landmarks as JSON, `POST /pose/annotate` returns the image with the skeleton drawn on. The model is wrapped behind a small estimator class and injected as a FastAPI dependency, which is what lets the tests run without the model file or GPU libraries. Details are in the [project README](fastapi-pose-detection/README.md).

## Where things stand

I'd rather be upfront about what has and hasn't been checked:

- Both projects pass their unit tests and lint in CI.
- The NPU backend's maths and its `ctypes` binding are tested (with synthetic tensors and a small C stand-in for the vendor library). Running it against the real driver and `.nb` file on the board is the next step, and I'll put measured numbers in the project README once I have them.
- The camera path (MIPI sensor through Allwinner's `vin_v4l2` driver) is likewise still to be confirmed on the board; `scripts/check_camera.py` is there for that.
- The pose API is a prototype: single image, one person, no authentication.

## Repository layout

```
.
├── orange-pi-object-detection/   # detection pipeline, CPU + NPU backends, native build script
├── fastapi-pose-detection/       # pose-detection API
├── Documents-Tools-Samples/      # board manuals and reference material (see below)
└── .github/workflows/            # one CI workflow per project (lint, tests, Docker build)
```

`Documents-Tools-Samples/` holds vendor documentation I use as a reference. The vendor SDKs I build against are not redistributed from this repo: the NPU project compiles Allwinner's helper library from your own copy of their `ai-sdk`.

## Running the tests

```bash
cd orange-pi-object-detection   # or fastapi-pose-detection
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Author

Iman Dashtpeyma, M.Sc. student at Stockholm University (DSV), software developer based in Sweden.

## License

MIT, see [LICENSE](LICENSE). Third-party models and SDKs keep their own licenses.

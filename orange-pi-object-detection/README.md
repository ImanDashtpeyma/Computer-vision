# Orange Pi Object Detection

Real-time object detection on an [Orange Pi 4 Pro](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-4-Pro.html) (Allwinner A733) with a 13MP OV13850 MIPI camera, with two interchangeable inference backends: a portable CPU one (pretrained MobileNet-SSD) and one that runs YOLOv5s on the board's NPU.

This isn't a from-scratch computer vision project -- both models are publicly available pretrained networks ([chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) and the YOLOv5s build shipped in Allwinner's ai-sdk). What I built is the pipeline around them: camera capture, a pluggable inference backend, the pre/post-processing and Python binding for the NPU path, a CLI, tests that run without any hardware attached, and a CI/Docker setup.

## Hardware

- Orange Pi 4 Pro, Allwinner A733 SoC, 3 TOPS NPU (VeriSilicon VIP9000)
- 13MP OV13850 MIPI camera module

## Two backends

| Backend | Model | Runs on | Status |
|---|---|---|---|
| `cpu` (default) | MobileNet-SSD (VOC, 20 classes) via OpenCV DNN | any Linux box with a camera | tested in CI |
| `npu` | YOLOv5s (COCO, 80 classes) as a prebuilt `.nb` from Allwinner's ai-sdk | the Orange Pi 4 Pro's NPU (VeriSilicon VIP9000, "NPU v3") | maths unit-tested; hardware path to be verified on the board |

Neither model was trained by me. The CPU backend exists so anyone can clone the repo and run it; the NPU backend is the reason the board is interesting.

### How the NPU backend works

Allwinner's `ai-sdk` ships a small C wrapper (`awnn_lib.c`) around the VIPLite driver (`libVIPhal.so`, SDK version 2.0 for the A733) with seven calls: init, create network from a `.nb` file, set input, run, get outputs, destroy, uninit. The vendor's own YOLOv5 demo is built on exactly these.

This repo does the same thing from Python:

1. `native/build_awnn.sh` compiles the vendor's `awnn_lib.c` into `native/libawnn_npu.so` **from your local copy of the SDK**. The vendor sources, headers and libraries are not redistributed here.
2. `src/awnn.py` loads that library with `ctypes`.
3. `src/yolov5.py` is a numpy port of the vendor demo's pre- and post-processing: letterbox to 640x640 uint8 RGB (CHW), then decode the three output tensors (strides 8/16/32), threshold, and class-agnostic NMS.
4. Only the network itself runs on the NPU.

```bash
# on the Orange Pi
sudo apt install libopencv-dev cmake gcc      # as in the Orange Pi manual, section 3.34
native/build_awnn.sh ~/ai-sdk                  # -> native/libawnn_npu.so
mkdir -p models
cp ~/ai-sdk/examples/yolov5/model/v3/yolov5.nb models/   # the v3 build is the one for the A733
python main.py --backend npu --device /dev/video0
```

Models other than the bundled YOLOv5s have to be converted with Allwinner's ACUITY toolkit (the manual describes a Docker image for that); this repo does not automate it.

**What is and isn't verified.** The pre/post-processing is covered by unit tests with synthetic tensors, and the ctypes binding is tested against a small C stand-in for the library. What CI cannot check is the real driver and `.nb` on the board; I run that by hand and will note measured speed here once I have numbers worth quoting.

### Camera

The OV13850 goes through Allwinner's `vin_v4l2` driver (`sudo modprobe vin_v4l2`); the manual lists `/dev/video0` and `/dev/video8` for the two MIPI connectors. Which node and pixel format OpenCV can actually read has to be checked on the board:

```bash
python scripts/check_camera.py
python main.py --device /dev/video8 --fourcc MJPG
```

The manual also ships a pybind11 `V4L2Camera` demo in `/opt/v4l2_opencv_demo`, which is the fallback if plain OpenCV capture doesn't work.

## Project layout

```
.
├── main.py                 # CLI entrypoint (capture -> detect -> show/save)
├── src/
│   ├── camera.py            # OpenCV VideoCapture wrapper
│   ├── backends.py           # InferenceBackend: OpenCVDNNBackend, NPUBackend
│   ├── yolov5.py             # YOLOv5 letterbox + box decoding/NMS (numpy)
│   ├── awnn.py               # ctypes binding for the vendor NPU helper library
│   ├── detector.py           # glue + drawing boxes/labels on frames
│   └── config.py             # camera/model/app configuration
├── native/build_awnn.sh      # builds libawnn_npu.so from your local ai-sdk (on the board)
├── scripts/                  # download_model.py (CPU weights), check_camera.py
├── tests/                    # run with no camera/model/hardware required
├── Dockerfile
└── .github/workflows/ci.yml  # lint (ruff) + tests (pytest) + docker build
```

## Setup

```bash
git clone <this-repo-url>
cd orange-pi-object-detection
pip install -r requirements.txt
python scripts/download_model.py   # fetches deploy.prototxt + mobilenet_iter_73000.caffemodel
```

On the Orange Pi, make sure the camera overlay is enabled and the device shows up:

```bash
v4l2-ctl --list-devices
```

## Running

```bash
python main.py --device /dev/video0
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--device` | Camera index or `/dev/videoN` path (default `0`) |
| `--backend` | `cpu` (default) or `npu` |
| `--nbg` / `--awnn-lib` | paths to the `.nb` file and `libawnn_npu.so` (npu backend) |
| `--fourcc` | camera pixel format, e.g. `MJPG` or `YUYV` |
| `--confidence` | Detection confidence threshold (default `0.5`) |
| `--no-window` | Don't open a GUI window (for headless/SSH use) |
| `--save PATH` | Save annotated output as an mp4 |

## Running with Docker

```bash
docker build -t orange-pi-object-detection .
docker run --device=/dev/video0 orange-pi-object-detection --no-window --save output/demo.mp4
```

## Tests & CI

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Tests use fake backends, synthetic tensors and a tiny C stub of the NPU library, so they run on any machine (including CI) without a camera, without the NPU, and without downloading model weights. GitHub Actions runs lint + tests + a Docker build on every push/PR -- see `.github/workflows/ci.yml`.

## Known limitations

- The NPU path is written against the vendor SDK's documented demo but has not been benchmarked here yet.
- NPU inference is YOLOv5s at a fixed 640x640 input; other models need conversion with ACUITY.
- The CPU model (MobileNet-SSD/VOC0712) only recognizes 20 classes; the NPU model (YOLOv5s/COCO) recognizes 80. Neither is open-vocabulary.
- No training was done here; the model is used as-is for inference.

## License

MIT -- see [LICENSE](LICENSE). The MobileNet-SSD weights are from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) (not trained by me); check that repo for its own licensing terms if you redistribute the weights.

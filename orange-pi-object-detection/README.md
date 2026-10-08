# Orange Pi Object Detection

Real-time object detection on an [Orange Pi 4 Pro](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-4-Pro.html) (Allwinner A733) with a 13MP OV13850 MIPI camera, built around a pretrained MobileNet-SSD model.

This isn't a from-scratch computer vision project -- the model is a well-known, publicly available pretrained network ([chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD), VOC0712 weights, mAP 0.727). What I built is the pipeline around it: camera capture tuned for this specific board/sensor, a pluggable inference backend, a CLI, tests that run without any hardware attached, and a CI/Docker setup that keeps all of that honest.

## Hardware

- Orange Pi 4 Pro, Allwinner A733 SoC, 3 TOPS NPU (VeriSilicon VIP9000)
- 13MP OV13850 MIPI camera module

## Why CPU, not NPU, is the default here

The A733 is a very new chip (announced late 2025), and its NPU software stack is still immature in public tooling as of this writing:

- ONNX Runtime currently only runs on CPU for this SoC -- a VIPLite execution provider for A733/T527 has been [proposed](https://github.com/microsoft/onnxruntime/issues/28244) but isn't shipped.
- The documented real path to the NPU is Allwinner's **ACUITY Toolkit**, which converts a model to their NBG format, run through the low-level `libVIPhal.so` (VIPLite) API -- not something exposed through a mainstream Python ML framework yet.

I *have* gotten the NPU working on this exact board, through Orange Pi/Allwinner's own official demo tooling -- so the hardware and vendor stack are real and functional. But wiring that board-specific, image-specific toolchain into a general-purpose, publicly cloneable repo isn't something I could do honestly without the real SDK in front of me for every step, and I'd rather ship a pipeline that actually works for anyone who clones this than one that pretends to.

So: `src/backends.py` has a `NPUBackend` class that documents exactly what the NPU path requires and raises `NotImplementedError` instead of faking it. The default and only *working* backend is `OpenCVDNNBackend`, which runs the same MobileNet-SSD model on CPU via OpenCV's DNN module. On the Orange Pi's 8-core A733, this is good for a live demo at a few frames per second -- not NPU speeds, but real, tested, and portable to any Linux box with a camera (not just this one board).

Finishing the NPU backend is the obvious next step once I'm back at the board with the ACUITY Toolkit installed.

## Project layout

```
.
├── main.py                 # CLI entrypoint (capture -> detect -> show/save)
├── src/
│   ├── camera.py            # OpenCV VideoCapture wrapper
│   ├── backends.py           # InferenceBackend: OpenCVDNNBackend (working), NPUBackend (stub)
│   ├── detector.py           # glue + drawing boxes/labels on frames
│   └── config.py             # camera/model/app configuration
├── scripts/download_model.py # fetches the pretrained weights (not committed to git)
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
| `--backend` | `cpu` (default, works) or `npu` (stub, raises) |
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

Tests mock the inference backend, so they run on any machine (including CI) without a camera, without the NPU, and without downloading the real model weights. GitHub Actions runs lint + tests + a Docker build on every push/PR -- see `.github/workflows/ci.yml`.

## Known limitations

- NPU acceleration is not implemented in this repo (see above) -- CPU inference only.
- MobileNet-SSD/VOC0712 only recognizes 20 object classes (people, vehicles, animals, household objects -- see `src/config.py` for the full list), not open-vocabulary detection.
- No training was done here; the model is used as-is for inference.

## License

MIT -- see [LICENSE](LICENSE). The MobileNet-SSD weights are from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) (not trained by me); check that repo for its own licensing terms if you redistribute the weights.

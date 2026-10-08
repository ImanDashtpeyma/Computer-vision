# FastAPI Pose Detection

A FastAPI service that detects human body pose (33 keypoints) in an uploaded image, built on [MediaPipe's Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Tasks API).

## What this is, and isn't

This is a generic, standalone pose-detection API: send it an image, get back landmark coordinates (or an annotated image). That's it.

It's also the technical core I'm extending for my master's thesis prototype, *"Feeling the Correction"* -- a Mixed Reality system that compares visual, auditory and vibrotactile feedback for at-home physiotherapy, supervised by Charles Windlin at Stockholm University (DSV). This repo deliberately stops at generic pose detection: no feedback/scoring logic, no participant data, no study results. That split is intentional -- thesis-specific code needs sign-off from my supervisor before anything related gets published, and keeping this repo generic means it can be shared freely regardless of where the thesis work stands.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/` | service info, points to `/docs` |
| POST | `/pose/detect` | JSON: 33 landmarks (x, y, z, visibility, presence), normalized 0-1 |
| POST | `/pose/annotate` | the input image with a skeleton overlay drawn on it (PNG) |

Interactive docs (Swagger UI) are available at `/docs` once the server is running.

## Project layout

```
.
├── app/
│   ├── main.py              # FastAPI app, route registration
│   ├── pose_estimation.py    # PoseEstimator: wraps mediapipe's PoseLandmarker
│   └── routers/
│       ├── health.py
│       └── pose.py           # /pose/detect, /pose/annotate
├── scripts/
│   ├── download_model.py     # fetches the pose_landmarker .task bundle
│   └── smoke_test.py         # manual real-model check (see below)
├── tests/                    # mock the model, run anywhere, no GPU/EGL needed
├── Dockerfile
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone <this-repo-url>
cd fastapi-pose-detection
pip install -r requirements.txt
python scripts/download_model.py   # fetches pose_landmarker_lite.task (~6MB)
```

MediaPipe's native bindings load `libEGL`/`libGL` even for CPU-only inference, so on a fresh Debian/Ubuntu machine you'll also need:

```bash
sudo apt install libegl1 libgl1 libglib2.0-0
```

## Running

```bash
uvicorn app.main:app --reload
```

Then, for example:

```bash
curl -X POST http://localhost:8000/pose/detect -F "file=@photo.jpg"
```

## Running with Docker

```bash
docker build -t fastapi-pose-detection .
docker run -p 8000:8000 fastapi-pose-detection
```

## Tests & CI

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Unit tests inject a fake `PoseEstimator` via FastAPI's dependency-override mechanism, so they test the API contract (request/response shape, error handling) without needing the real model file or system GL libraries -- which is also why they're fast and safe to run in CI as-is.

For an actual end-to-end check against the real MediaPipe model, run:

```bash
python scripts/smoke_test.py path/to/a/photo.jpg
```

This isn't part of the automated test suite on purpose -- it needs the model downloaded and the system libraries above installed, neither of which a CI runner should need just to validate the API logic.

GitHub Actions (`.github/workflows/ci.yml`) runs lint + unit tests + a Docker build on every push/PR.

## Known limitations

- Single-image detection only (no video/stream endpoint yet).
- One pose per image is returned (MediaPipe's `IMAGE` running mode, single-person).
- No authentication/rate limiting -- this is a portfolio/prototype service, not production-hardened.
- The `PoseEstimator` holds one landmarker instance behind a lock, so concurrent requests are serialized rather than parallelized; fine for a demo, not for high-throughput production use.

## License

MIT -- see [LICENSE](LICENSE).

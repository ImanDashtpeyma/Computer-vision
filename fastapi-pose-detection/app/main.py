from fastapi import FastAPI

from app.routers import health, pose

app = FastAPI(
    title="Pose Detection API",
    description=(
        "A generic body-movement/pose detection service built on MediaPipe's "
        "Pose Landmarker. This is the reusable core later extended for a "
        "Mixed Reality physiotherapy-feedback thesis prototype -- this repo "
        "intentionally stops at generic pose detection and contains no "
        "thesis-specific feedback logic or participant data."
    ),
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(pose.router)


@app.get("/")
def root() -> dict:
    return {"service": "pose-detection-api", "docs": "/docs"}

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.pose_estimation import POSE_CONNECTIONS, PoseEstimator, get_pose_estimator

router = APIRouter(prefix="/pose", tags=["pose"])


def _decode_upload_to_rgb(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Supported: JPEG, PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@router.post("/detect")
async def detect_pose(
    file: UploadFile = File(...),
    estimator: PoseEstimator = Depends(get_pose_estimator),
) -> dict:
    """Run pose detection on a single uploaded image, return raw landmarks."""
    rgb = _decode_upload_to_rgb(await file.read())
    result = estimator.detect(rgb)
    return {
        "pose_detected": result.pose_detected,
        "landmarks": [lm.__dict__ for lm in result.landmarks],
    }


@router.post("/annotate")
async def annotate_pose(
    file: UploadFile = File(...),
    estimator: PoseEstimator = Depends(get_pose_estimator),
) -> Response:
    """Run pose detection and return the image with a skeleton overlay drawn on it."""
    rgb = _decode_upload_to_rgb(await file.read())
    result = estimator.detect(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]

    if result.pose_detected:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in result.landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(bgr, points[a], points[b], (0, 200, 0), 2)
        for x, y in points:
            cv2.circle(bgr, (x, y), 3, (0, 120, 255), -1)

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return Response(content=encoded.tobytes(), media_type="image/png")

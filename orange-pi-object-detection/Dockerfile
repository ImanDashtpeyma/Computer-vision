# CPU-only image. This deliberately does NOT try to package Allwinner's NPU
# stack (ACUITY Toolkit / libVIPhal.so) -- that's vendor/board-specific and
# documented in the README instead of being faked here. Build/run this on
# the Orange Pi itself (aarch64) or any Linux amd64/arm64 box with a camera.
FROM python:3.11-slim

# Runtime libraries OpenCV needs that python:slim doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the pretrained MobileNet-SSD weights at build time so the image is
# ready to run out of the box.
RUN python scripts/download_model.py

# No camera inside a container by default -- pass one through explicitly:
#   docker run --device=/dev/video0 orange-pi-object-detection --no-window --save output/demo.mp4
ENTRYPOINT ["python", "main.py"]
CMD ["--no-window"]

FROM python:3.11-slim

# System libraries mediapipe and opencv need at runtime that python:slim
# doesn't ship (libEGL/libGL -- mediapipe's native bindings dlopen these
# even for CPU-only inference).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the pose landmarker model at build time so the image works out of the box.
RUN python scripts/download_model.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

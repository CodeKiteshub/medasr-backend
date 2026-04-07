# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model weights at build time so the first request is fast.
# This adds ~420MB to the image but avoids a 60s cold download on first request.
RUN python - <<'EOF'
from transformers import pipeline
pipeline("automatic-speech-recognition", model="google/medasr", device=-1)
print("Model cached.")
EOF

# ── Stage 2: app ──────────────────────────────────────────────────────────────
COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]

"""
MedASR FastAPI backend — wraps google/medasr (HuggingFace) via transformers pipeline.

Endpoint:
  POST /transcribe
  Body: multipart/form-data, field "file" = raw PCM 16kHz 16-bit mono audio bytes

Response:
  { "transcript": "..." }

Required Railway environment variable:
  HF_TOKEN — HuggingFace read token (model is gated, requires licence acceptance).
  Get one at: https://huggingface.co/settings/tokens
  Also accept the model licence at: https://huggingface.co/google/medasr

Runs CPU-only. Inference time: ~15-30s per clip on CPU.
Model is downloaded on first startup (~420MB) and cached in /tmp/hf_cache.
"""

import logging
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Point HF cache to /tmp so it persists within the container session.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# transformers / huggingface_hub automatically reads HUGGINGFACE_HUB_TOKEN.
# We just copy HF_TOKEN → HUGGINGFACE_HUB_TOKEN so Railway's variable name works.
hf_token = os.environ.get("HF_TOKEN", "").strip()
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    logger.info("HF_TOKEN found — will authenticate with HuggingFace.")
else:
    logger.warning("HF_TOKEN not set — gated model download will fail.")

app = FastAPI(title="MedASR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup.
# First boot downloads ~420MB weights — takes 60-120s depending on Railway bandwidth.
# Subsequent restarts use /tmp/hf_cache (cache survives container restarts on Railway).
logger.info("Loading google/medasr model…")
_pipe = pipeline(
    "automatic-speech-recognition",
    model="google/medasr",
    device=-1,  # CPU; change to 0 for GPU
    token=hf_token if hf_token else None,
    trust_remote_code=True,  # required: model uses custom lasr_ctc architecture
)
logger.info("Model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accepts raw PCM 16kHz 16-bit mono audio bytes and returns the transcript.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Convert raw PCM bytes → float32 numpy array normalised to [-1, 1]
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    if audio_float32.size == 0:
        raise HTTPException(status_code=400, detail="No audio samples in file")

    logger.info(
        "Transcribing %d samples (%.1fs)…",
        audio_float32.size,
        audio_float32.size / 16000,
    )

    try:
        result = _pipe(
            {"raw": audio_float32, "sampling_rate": 16000},
            return_timestamps=False,
        )
        transcript: str = result.get("text", "").strip()  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Transcript: %s", transcript[:120])
    return {"transcript": transcript}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

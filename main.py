"""
MedASR FastAPI backend — wraps google/medasr (HuggingFace) via transformers pipeline.

Endpoint:
  POST /transcribe
  Body: multipart/form-data, field "file" = raw PCM 16kHz 16-bit mono audio bytes

Response:
  { "transcript": "..." }

Runs CPU-only (Railway free tier / Pro without GPU).
Inference time: ~15-30s per audio clip on CPU, ~2-5s with GPU.
"""

import io
import logging
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedASR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup — takes ~30-60s on first boot (downloads ~420MB weights)
logger.info("Loading google/medasr model…")
_pipe = pipeline(
    "automatic-speech-recognition",
    model="google/medasr",
    device=-1,  # CPU; change to 0 for GPU
)
logger.info("Model loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accepts raw PCM 16kHz 16-bit mono audio bytes and returns the transcript.
    The Flutter app sends audio recorded at 16 kHz, LINEAR16 (int16), mono.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Convert raw PCM bytes → float32 numpy array normalised to [-1, 1]
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    if audio_float32.size == 0:
        raise HTTPException(status_code=400, detail="No audio samples in file")

    logger.info("Transcribing %d samples (%.1fs)…", audio_float32.size, audio_float32.size / 16000)

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

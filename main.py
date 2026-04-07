"""
MedASR FastAPI backend — wraps google/medasr (HuggingFace) for medical ASR.

Endpoint:
  POST /transcribe
  Body: multipart/form-data, field "file" = raw PCM 16kHz 16-bit mono audio bytes

Response:
  { "transcript": "..." }

Required Railway environment variable:
  HF_TOKEN — HuggingFace read token (model is gated, requires licence acceptance).
  Get one at: https://huggingface.co/settings/tokens
  Also accept the model licence at: https://huggingface.co/google/medasr

Requires transformers>=5.0.0 (installed from git — lasr_ctc arch not in 4.x).
Runs CPU-only. Inference time: ~15-30s per clip on CPU.
"""

import logging
import os
import tempfile
import wave

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCTC, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── HuggingFace auth ──────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

hf_token = os.environ.get("HF_TOKEN", "").strip()
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    logger.info("HF_TOKEN found — will authenticate with HuggingFace.")
else:
    logger.warning("HF_TOKEN not set — gated model download will fail.")

# ── Load model and processor at startup ───────────────────────────────────────
MODEL_ID = "google/medasr"
SAMPLE_RATE = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Loading %s on %s…", MODEL_ID, device)
processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token or None)
model = AutoModelForCTC.from_pretrained(MODEL_ID, token=hf_token or None).to(device)
model.eval()
logger.info("Model loaded and ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="MedASR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


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

    # Convert raw PCM int16 bytes → float32 numpy array normalised to [-1, 1]
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    if audio_float32.size == 0:
        raise HTTPException(status_code=400, detail="No audio samples in file")

    duration = audio_float32.size / SAMPLE_RATE
    logger.info("Transcribing %d samples (%.1fs)…", audio_float32.size, duration)

    try:
        # Prepare inputs
        inputs = processor(
            audio_float32,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model.generate(**inputs)

        transcript = processor.batch_decode(outputs)[0].strip()
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Transcript: %s", transcript[:120])
    return {"transcript": transcript}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

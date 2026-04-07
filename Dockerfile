FROM python:3.11-slim

WORKDIR /app

# System deps for torch CPU, librosa (needs libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (includes transformers from git for >=5.0 with lasr_ctc support)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Model weights downloaded at startup using HF_TOKEN env var set in Railway Variables.
CMD ["python", "main.py"]

FROM python:3.11-slim

WORKDIR /app

# System deps for torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Model weights are downloaded at first startup using HF_TOKEN env var.
# Set HF_TOKEN in Railway → Variables before deploying.
CMD ["python", "main.py"]

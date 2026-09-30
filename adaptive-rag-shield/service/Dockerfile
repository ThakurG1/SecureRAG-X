# CUDA-enabled base image -- required for local Llama inference (bitsandbytes needs the CUDA toolkit
# headers at build time). If you truly want CPU-only, switch to python:3.11-slim and set
# HF_LOAD_4BIT=0 in the environment (expect Llama-3-8B to be very slow and need ~16-20GB RAM).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptive_rag_shield.py /adaptive_rag_shield.py
COPY service/ .

ENV LLM_BACKEND=hf \
    HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
    HF_LOAD_4BIT=1 \
    RATE_LIMIT_PER_MIN=30 \
    PORT=8000
    # HF_TOKEN must be passed at `docker run -e HF_TOKEN=...` time, not baked into the image

EXPOSE 8000
# --workers 1: the model is loaded once per process; a second worker would double VRAM use.
# Scale horizontally (more GPU replicas) instead -- see README.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

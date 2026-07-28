# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# ---- System dependencies -----------------------------------------------------------
# - tesseract-ocr: enables the OCR fallback path for scanned PDFs (app runs fine
#   without it too -- OCR is simply reported unavailable if this layer is removed).
# - libgl1 / libglib2.0-0: runtime libs some PDF/image/ML wheels expect on slim images.
# - curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (layer-cached separately from application code) ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code --------------------------------------------------------------
COPY . .

# ---- Non-root runtime user ---------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/raw_documents /app/data/vector_db /app/data/dataset \
               /app/data/extracted_images /app/models \
    && chown -R appuser:appuser /app

USER appuser

# Model downloads (embeddings, reranker) and Hugging Face cache persist here.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form (not exec-form JSON array) so ${PORT:-8000} is expanded at
# container start -- this lets the same image run locally on 8000 via
# docker-compose AND on Render, which injects its own $PORT value.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

# SwiftCFD inference service — CPU only, slim base.
#
#   docker build -t swiftcfd-service .
#   docker run --rm -p 8000:8000 swiftcfd-service
#
# Weights are downloaded from the Hugging Face Hub on first start and cached
# under /cache/hf (mount it as a volume to avoid re-downloading).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    SWIFTCFD_WEIGHTS=hf \
    SWIFTCFD_TORCH_THREADS=1

WORKDIR /app

# CPU-only torch wheel keeps the image at roughly a fifth of the CUDA build.
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r service/requirements.txt

COPY service/ ./service/

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache/hf /app/logs \
    && chown -R app:app /cache /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

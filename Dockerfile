# IndexTTS-2.5 FastAPI service image
#
# Design principle: the image contains only [code + dependencies], not model weights.
# Models are mounted at runtime via volumes (see MODEL_DIR in docker-compose.yml).
#
# Build (from the project root):
#   docker build -t indextts-api .
#   # Optional: pin the upstream version
#   docker build -t indextts-api --build-arg INDEXTTS_REF=v2.5 .
#
# Run:
#   docker compose up -d
#   # Or manually:
#   docker run -d --gpus all -p 8000:8000 \
#     -v /path/to/checkpoints:/models/checkpoints \
#     -e MODEL_DIR=/models/checkpoints indextts-api
#   (mount writable so missing model weights can be auto-downloaded on first start)

# CUDA 12.8 base image (matches torch 2.8+cu128)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Build-time proxy (passed via --build-arg, used by curl/git/uv for downloads).
# Direct connections to astral.sh / github.com / pypi.org may fail on China networks;
# recommend passing the host proxy:
#   --build-arg http_proxy=http://<host>:<port> --build-arg https_proxy=http://<host>:<port>
ARG http_proxy
ARG https_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default model directory (can be overridden at runtime with -e MODEL_DIR)
    MODEL_DIR=/models/checkpoints

# System dependencies: python3.11 + basic tools
# Note: Ubuntu 24.04 (noble) official repos only have python3.12; the project requires <3.12,
# so python3.11 is installed via the deadsnakes PPA.
# apt repos (ubuntu/launchpad) work with direct connections; using a proxy causes SSL failures,
# so the proxy is explicitly unset.
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY && \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install uv (official standalone installer, no pip dependency; matches the project's uv.lock version)
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh \
    && uv --version

WORKDIR /app

# ---------------------------------------------------------------------------
# 1) Clone index-tts source from the upstream repo (image does not include models)
# ---------------------------------------------------------------------------
ARG INDEXTTS_REF=main
RUN git clone --depth 1 --branch ${INDEXTTS_REF} \
        https://github.com/index-tts/index-tts.git /app \
    && rm -rf /app/.git

# ---------------------------------------------------------------------------
# 2) Install dependencies (core + webui extra + FastAPI service deps)
#    torch/torchaudio installed from the pytorch cu128 index (configured in pyproject)
# ---------------------------------------------------------------------------
# uv sync creates a virtual environment at /app/.venv and installs all dependencies (incl. webui extra);
# the runtime CMD uses /app/.venv/bin/python
RUN uv sync --frozen --extra webui \
    && uv pip install --python /app/.venv/bin/python fastapi uvicorn python-multipart

# ---------------------------------------------------------------------------
# 3) Copy this project's API service (thin wrapper, only depends on the upstream indextts package)
# ---------------------------------------------------------------------------
COPY api_server.py /app/api_server.py
COPY test.html /app/test.html

# Model + voices mount points (empty directory placeholders, overridden by volumes at runtime)
RUN mkdir -p /models/checkpoints /models/voices
VOLUME ["/models"]

EXPOSE 8000

# Clear proxy at runtime (build-time proxy is only for downloads; in-container localhost access should not go through a proxy)
ENV http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY=

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1

# Start the API service by default; can be overridden with docker run
# Optional flags: --bf16 --qwen_emo --accel --deepspeed --torch_compile
CMD ["/app/.venv/bin/python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]

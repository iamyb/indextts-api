# IndexTTS-2.5 FastAPI 服务镜像
#
# 设计原则：镜像只包含【代码 + 依赖】，不包含模型权重。
# 模型在运行时通过 volume 挂载（见 docker-compose.yml 的 MODEL_DIR）。
#
# 构建（在项目根目录）：
#   docker build -t indextts-api .
#   # 可选：锁定上游版本
#   docker build -t indextts-api --build-arg INDEXTTS_REF=v2.5 .
#
# 运行：
#   docker compose up -d
#   # 或手动：
#   docker run -d --gpus all -p 8000:8000 \
#     -v /path/to/checkpoints:/models/checkpoints:ro \
#     -e MODEL_DIR=/models/checkpoints indextts-api

# CUDA 12.8 基础镜像（与 torch 2.8+cu128 匹配）
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# 构建期代理（通过 --build-arg 传入，供 curl/git/uv 下载使用）。
# 国内网络直连 astral.sh / github.com / pypi.org 可能失败，建议传宿主机代理：
#   --build-arg http_proxy=http://<host>:<port> --build-arg https_proxy=http://<host>:<port>
ARG http_proxy
ARG https_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 默认模型目录（运行时可被 -e MODEL_DIR 覆盖）
    MODEL_DIR=/models/checkpoints

# 系统依赖：python3.11 + 基础工具
# 注意：Ubuntu 24.04 (noble) 官方源只有 python3.12，项目要求 <3.12，
# 所以通过 deadsnakes PPA 安装 python3.11。
# apt 源（ubuntu/launchpad）直连即可，走代理反而 SSL 失败，故显式取消代理。
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY && \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（官方独立安装脚本，不依赖 pip；与项目 uv.lock 版本一致）
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh \
    && uv --version

WORKDIR /app

# ---------------------------------------------------------------------------
# 1) 从上游仓库拉取 index-tts 代码（镜像不包含模型）
# ---------------------------------------------------------------------------
ARG INDEXTTS_REF=main
RUN git clone --depth 1 --branch ${INDEXTTS_REF} \
        https://github.com/index-tts/index-tts.git /app \
    && rm -rf /app/.git

# ---------------------------------------------------------------------------
# 2) 安装依赖（核心 + webui extra + FastAPI 服务依赖）
#    torch/torchaudio 从 pytorch cu128 索引安装（pyproject 已配置）
# ---------------------------------------------------------------------------
# uv sync 在 /app/.venv 创建虚拟环境并安装全部依赖（含 webui extra）；
# 运行时 CMD 使用 /app/.venv/bin/python
RUN uv sync --frozen --extra webui \
    && uv pip install --python /app/.venv/bin/python fastapi uvicorn python-multipart

# ---------------------------------------------------------------------------
# 3) 放入本项目的 API 服务（薄封装，只依赖上游 indextts 包）
# ---------------------------------------------------------------------------
COPY api_server.py /app/api_server.py

# 模型 + 音色挂载点（空目录占位，运行时由 volume 覆盖）
RUN mkdir -p /models/checkpoints /models/voices
VOLUME ["/models"]

EXPOSE 8000

# 运行时清空代理（构建期代理仅用于下载；容器内访问 localhost 服务不应走代理）
ENV http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY=

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1

# 默认启动 API 服务；可用 docker run 覆盖命令
# 可选参数: --bf16 --qwen_emo --accel --deepspeed --torch_compile
CMD ["/app/.venv/bin/python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]

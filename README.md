# IndexTTS-2.5 API Server

Wraps [IndexTTS-2.5](https://github.com/index-tts/index-tts) as a **FastAPI HTTP service**,
providing native REST endpoints and **OpenAI-compatible** `/v1/audio/speech` + voice management endpoints, with one-click Docker deployment.

> This repo is a **thin wrapper**: it only contains the API service code and Docker deployment files.
> The upstream index-tts source code and model weights are **not in this repo** —
> the source is automatically `git clone`d during Docker build, and models are mounted at runtime via volumes.

## Features

- **Native endpoints**: `/tts` (multipart), `/tts/json`, `/tts/stream` (SSE streaming)
- **OpenAI-compatible**: `POST /v1/audio/speech`, directly compatible with the OpenAI SDK ecosystem
- **Voice management**: `GET/POST/DELETE /v1/voices` — upload reference audio to clone a voice
- **Model-specific parameters**: passed via `extra_params` (emotion vector, language, text normalization, etc.)
- **Docker deployment**: pre-built images on [GHCR](https://ghcr.io/iamyb/indextts-api) (just `docker pull`); image contains only code + dependencies, models are mounted at runtime
- **GPU serialization**: single-GPU inference is serialized with a lock to prevent concurrent OOM

## Directory Structure

```
.
├── api_server.py        # FastAPI service (core)
├── Dockerfile           # Build image (auto-clones upstream index-tts)
├── docker-compose.yml   # One-click deployment
├── .dockerignore        # Build context exclusions
└── README.md
```

## Prerequisites

1. **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   (IndexTTS-2.5 recommends 12GB+ VRAM)
2. **Docker** + Docker Compose
3. **Model weights** (optional — auto-downloaded on first start): the server checks
   `./checkpoints/` for the primary files (`gpt.pth`, `s2mel.pth`, `codec.pth`, `tiktoken`,
   `wav2vec2bert_stats.pt`) and, if any are missing, automatically downloads the
   `IndexTeam/IndexTTS-2.5` repo (HuggingFace, or ModelScope/hf-mirror in China) plus the
   auxiliary models into `hf_cache/` (w2v-bert-2.0, semantic_codec_model.safetensors,
   campplus_cn_common.bin, bigvgan/). The download is ~11GB and the mount must be
   **writable** (no `:ro`). Set `AUTO_DOWNLOAD_MODELS=0` to disable auto-download and
   require pre-downloaded weights.

## Quick Start (Pre-built Image)

Pre-built images are published to [GitHub Container Registry](https://ghcr.io) on every release —
no local build needed:

```bash
# 1. Pull and run (image is ~14GB, first pull may take a while)
#    On first start the server auto-downloads ~11GB of model weights into ./checkpoints/
#    (the mount is writable on purpose — remove :ro if you pre-download the weights)
docker pull ghcr.io/iamyb/indextts-api:latest
docker run -d --name indextts-api --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/models/checkpoints \
  -v $(pwd)/voices:/models/voices \
  ghcr.io/iamyb/indextts-api:latest

# 2. Check logs (first start: model download + loading may take a while)
docker logs -f indextts-api

# 3. Health check
curl http://127.0.0.1:8000/health
# => {"status":"ok","model_loaded":true,"device":"cuda:0","gpu":true}
```

Pinned versions are also available, e.g. `ghcr.io/iamyb/indextts-api:0.1.1`.

> **China network**: `ghcr.io` may be slow or unreachable. Options:
> - use a GHCR mirror/proxy for `docker pull`, or
> - build locally with a proxy (see [Building from Source](#building-from-source)).

## Building from Source

If you need to build the image yourself (e.g. to pin a specific upstream commit):

```bash
# First build clones upstream code + installs deps, may be slow
docker compose up -d --build

# Pin the upstream index-tts version (branch/tag/commit), defaults to main
INDEXTTS_REF=v2.5 docker compose up -d --build
```

> **China network**: downloading astral.sh / github / pypi during build may require a proxy. Pass via environment variables:
> ```bash
> HTTP_PROXY=http://<host>:<port> HTTPS_PROXY=http://<host>:<port> docker compose build
> ```
> At runtime, the container does not use a proxy (compose clears them).

### Releasing a New Image

Push a `v*` tag and GitHub Actions builds + pushes to GHCR automatically:

```bash
git tag v0.1.2 && git push origin v0.1.2
```

The upstream index-tts version used by the CI build is controlled by the repo
variable `INDEXTTS_REF` (Settings → Variables → Actions), defaulting to `main`.

## API Endpoints

### Native Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/tts` | multipart: `text` + `ref_audio` (upload) + `lang` → WAV |
| `POST` | `/tts/json` | JSON body: `ref_audio` via local path → WAV |
| `POST` | `/tts/stream` | SSE streaming with base64 audio chunks |

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -F "text=Hello" -F "ref_audio=@voice_01.wav" -F "lang=ZH" -o out.wav
```

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/speech` | Text-to-speech → 16-bit PCM WAV |
| `GET` | `/v1/voices` | List voices |
| `POST` | `/v1/voices` | Upload a voice (requires `consent` declaration) |
| `DELETE` | `/v1/voices/{voice_id}` | Delete a voice |

**`/v1/audio/speech` request body:**

```json
{
  "model": "indextts",
  "input": "Hello, this is the OpenAI-compatible endpoint.",
  "voice": "voice_01",
  "response_format": "wav",
  "speed": 1.0,
  "extra_params": {
    "lang": "zh",
    "text_normalization": true
  }
}
```

- `speed`: range `[0.5, 2.0]`, out of range returns 422; maps to `duration_factor = 1/speed` (native speed change, no resampling)
- `extra_params`: model-specific parameters, takes priority over flat fields. Supported keys:
  `lang`, `text_normalization`, `emo_vector` (8-dim emotion vector), `emo_alpha`,
  `emo_text`, `use_emo_text`, `emo_audio`, `use_random`

```bash
curl -s -o out.wav -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello","voice":"voice_01","extra_params":{"lang":"zh"}}'
```

**Voice management:**

```bash
# Upload (consent is required — voice cloning consent declaration)
curl -X POST http://127.0.0.1:8000/v1/voices \
  -F "file=@voice_01.wav" -F "name=my_voice" \
  -F "consent=I confirm I have the right to use this voice"

# List
curl http://127.0.0.1:8000/v1/voices

# Delete
curl -X DELETE http://127.0.0.1:8000/v1/voices/my_voice
```

Voice ID = filename without extension; only alphanumeric, `-`, `_` are allowed.

## Running Without Docker

```bash
# In the index-tts source directory, after installing dependencies via conda/venv:
env -u http_proxy -u https_proxy -u all_proxy \
  USE_MODELSCOPE=false HF_ENDPOINT=https://hf-mirror.com \
  python api_server.py --port 8000 --model_dir ./checkpoints
```

Optional flags: `--bf16`, `--qwen_emo`, `--accel`, `--deepspeed`, `--torch_compile`, `--cuda_kernel`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/models/checkpoints` | Model weights directory |
| `AUTO_DOWNLOAD_MODELS` | `1` | Auto-download missing model weights on startup (`0` to disable) |
| `VOICES_DIR` | `voices` | Voice reference audio directory |
| `INDEXTTS_REF` | `main` | Upstream branch/tag/commit to clone at build time |
| `HTTP_PROXY` / `HTTPS_PROXY` | empty | Build-time proxy (only needed during build) |

## License

The API wrapper code in this repo follows the upstream [IndexTTS](https://github.com/index-tts/index-tts) license terms.
Model weights are subject to their respective license agreements.

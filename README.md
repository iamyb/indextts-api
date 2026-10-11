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
- **Docker deployment**: image contains only code + dependencies; models are mounted at runtime; upstream code is auto-cloned at build time
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
3. **Model weights**: place in the host `./checkpoints/` directory, containing:
   - `gpt.pth`, `s2mel.pth`, `codec.pth`, `tiktoken`, `wav2vec2bert_stats.pt`
   - `hf_cache/` (w2v-bert-2.0, semantic_codec_model.safetensors, campplus_cn_common.bin, bigvgan/)

   Models can be downloaded from HuggingFace (use `https://hf-mirror.com` mirror in China).

## Quick Start (Docker)

```bash
# 1. Place model weights in ./checkpoints/ (see above)
# 2. Build and start (first build clones upstream code + installs deps, may be slow)
docker compose up -d --build

# 3. Check logs (model loading takes ~1-2 minutes)
docker compose logs -f

# 4. Health check
curl http://127.0.0.1:8000/health
# => {"status":"ok","model_loaded":true,"device":"cuda:0","gpu":true}
```

> **China network**: downloading astral.sh / github / pypi during build may require a proxy. Pass via environment variables:
> ```bash
> HTTP_PROXY=http://<host>:<port> HTTPS_PROXY=http://<host>:<port> docker compose build
> ```
> At runtime, the container does not use a proxy (compose clears them).

### Pinning the Upstream Version

By default, the `main` branch is built. To pin a version, set `INDEXTTS_REF`:

```bash
INDEXTTS_REF=v2.5 docker compose up -d --build
```

### Published Images (GHCR)

Pre-built images are published to [GitHub Container Registry](https://ghcr.io) on every `v*` tag
(via GitHub Actions, no local build needed):

```bash
# Pull and run a published image
docker pull ghcr.io/iamyb/indextts-api:latest
docker run -d --gpus all -p 8000:8000 \
  -v /path/to/checkpoints:/models/checkpoints:ro \
  -v /path/to/voices:/models/voices \
  ghcr.io/iamyb/indextts-api:latest
```

To release a new version, just push a tag:

```bash
git tag v0.1.1 && git push origin v0.1.1
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
| `VOICES_DIR` | `voices` | Voice reference audio directory |
| `INDEXTTS_REF` | `main` | Upstream branch/tag/commit to clone at build time |
| `HTTP_PROXY` / `HTTPS_PROXY` | empty | Build-time proxy (only needed during build) |

## License

The API wrapper code in this repo follows the upstream [IndexTTS](https://github.com/index-tts/index-tts) license terms.
Model weights are subject to their respective license agreements.

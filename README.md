# IndexTTS-2.5 API Server

把 [IndexTTS-2.5](https://github.com/index-tts/index-tts) 封装成一个 **FastAPI HTTP 服务**，
提供原生 REST 接口和 **OpenAI 兼容** 的 `/v1/audio/speech` + 音色管理接口，支持 Docker 一键部署。

> 本仓库是**薄封装**：只包含 API 服务代码和 Docker 部署文件。
> 上游 index-tts 源码和模型权重**都不在仓库里**——
> 源码在 Docker 构建时自动 `git clone`，模型在运行时通过 volume 挂载。

## 特性

- **原生接口**：`/tts`（multipart）、`/tts/json`、`/tts/stream`（SSE 流式）
- **OpenAI 兼容**：`POST /v1/audio/speech`，可直接对接 OpenAI SDK 生态
- **音色管理**：`GET/POST/DELETE /v1/voices`，上传参考音频即可克隆音色
- **模型专属参数**：通过 `extra_params` 传入（情感向量、语言、文本归一化等）
- **Docker 部署**：镜像只含代码+依赖，模型运行时挂载，构建时自动拉取上游代码
- **GPU 串行化**：单卡推理用锁串行，防并发 OOM

## 目录结构

```
.
├── api_server.py        # FastAPI 服务（核心）
├── Dockerfile           # 构建镜像（自动 clone 上游 index-tts）
├── docker-compose.yml   # 一键部署
├── .dockerignore        # 构建上下文排除项
└── README.md
```

## 前置条件

1. **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   （IndexTTS-2.5 建议 12GB+ 显存）
2. **Docker** + Docker Compose
3. **模型权重**：放到宿主机 `./checkpoints/` 目录，包含：
   - `gpt.pth`、`s2mel.pth`、`codec.pth`、`tiktoken`、`wav2vec2bert_stats.pt`
   - `hf_cache/`（w2v-bert-2.0、semantic_codec_model.safetensors、campplus_cn_common.bin、bigvgan/）

   模型可从 HuggingFace 下载（国内可用 `https://hf-mirror.com` 镜像）。

## 快速开始（Docker）

```bash
# 1. 把模型权重放到 ./checkpoints/（见上）
# 2. 构建并启动（首次构建会 clone 上游代码 + 装依赖，较慢）
docker compose up -d --build

# 3. 查看日志（模型加载约 1-2 分钟）
docker compose logs -f

# 4. 健康检查
curl http://127.0.0.1:8000/health
# => {"status":"ok","model_loaded":true,"device":"cuda:0","gpu":true}
```

> **国内网络**：构建期下载 astral.sh / github / pypi 可能需要代理，通过环境变量传入：
> ```bash
> HTTP_PROXY=http://<host>:<port> HTTPS_PROXY=http://<host>:<port> docker compose build
> ```
> 运行时容器内不走代理（compose 已清空）。

### 锁定上游版本

默认构建 `main` 分支。要锁定版本，设置 `INDEXTTS_REF`：

```bash
INDEXTTS_REF=v2.5 docker compose up -d --build
```

## API 接口

### 原生接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/tts` | multipart：`text` + `ref_audio`（上传）+ `lang` → WAV |
| `POST` | `/tts/json` | JSON 体：`ref_audio` 用本地路径 → WAV |
| `POST` | `/tts/stream` | SSE 流式返回 base64 音频分块 |

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -F "text=你好" -F "ref_audio=@voice_01.wav" -F "lang=ZH" -o out.wav
```

### OpenAI 兼容接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/audio/speech` | 文本转语音 → 16-bit PCM WAV |
| `GET` | `/v1/voices` | 列出音色 |
| `POST` | `/v1/voices` | 上传音色（需 `consent` 授权声明） |
| `DELETE` | `/v1/voices/{voice_id}` | 删除音色 |

**`/v1/audio/speech` 请求体：**

```json
{
  "model": "indextts",
  "input": "你好，这是 OpenAI 兼容接口。",
  "voice": "voice_01",
  "response_format": "wav",
  "speed": 1.0,
  "extra_params": {
    "lang": "zh",
    "text_normalization": true
  }
}
```

- `speed`：范围 `[0.5, 2.0]`，越界返回 422；映射为 `duration_factor = 1/speed`（原生变速，不重采样）
- `extra_params`：模型专属参数，优先级高于扁平字段。支持键：
  `lang`、`text_normalization`、`emo_vector`（8 维情感向量）、`emo_alpha`、
  `emo_text`、`use_emo_text`、`emo_audio`、`use_random`

```bash
curl -s -o out.wav -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"voice_01","extra_params":{"lang":"zh"}}'
```

**音色管理：**

```bash
# 上传（consent 必填，音色克隆授权声明）
curl -X POST http://127.0.0.1:8000/v1/voices \
  -F "file=@voice_01.wav" -F "name=my_voice" \
  -F "consent=I confirm I have the right to use this voice"

# 列出
curl http://127.0.0.1:8000/v1/voices

# 删除
curl -X DELETE http://127.0.0.1:8000/v1/voices/my_voice
```

音色 id = 文件名去扩展名，只允许字母数字 `-` `_`。

## 不依赖 Docker 直接运行

```bash
# 在 index-tts 源码目录，用 conda/venv 装好依赖后：
env -u http_proxy -u https_proxy -u all_proxy \
  USE_MODELSCOPE=false HF_ENDPOINT=https://hf-mirror.com \
  python api_server.py --port 8000 --model_dir ./checkpoints
```

可选参数：`--bf16`、`--qwen_emo`、`--accel`、`--deepspeed`、`--torch_compile`、`--cuda_kernel`。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_DIR` | `/models/checkpoints` | 模型权重目录 |
| `VOICES_DIR` | `voices` | 音色参考音频目录 |
| `INDEXTTS_REF` | `main` | 构建时 clone 的上游分支/tag/commit |
| `HTTP_PROXY` / `HTTPS_PROXY` | 空 | 构建期代理（仅构建时需要） |

## 许可

本仓库的 API 封装代码遵循上游 [IndexTTS](https://github.com/index-tts/index-tts) 的许可条款。
模型权重请遵守其各自的许可协议。

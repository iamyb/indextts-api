"""IndexTTS-2.5 FastAPI HTTP 服务。

把 IndexTTS2 的推理能力暴露成 REST API，方便其他程序通过 HTTP 调用。

启动（注意：本地服务必须取消代理，否则 gradio/uvicorn 自检会走代理失败）：

    cd /home/iamyb/github/index-tts
    env -u http_proxy -u https_proxy -u all_proxy \
        USE_MODELSCOPE=false HF_ENDPOINT=https://hf-mirror.com \
        /home/iamyb/miniconda3/envs/indextts/bin/python api_server.py --port 8000

接口：
    GET  /health          健康检查
    POST /tts             文本 + 参考音频 -> WAV 音频
    POST /tts/json        JSON 方式合成（参考音频用本地路径）
    POST /tts/stream      同上，但用 SSE 流式返回音频分块（低首包延迟）
    POST /v1/audio/speech OpenAI 兼容合成（voice 映射到 voices 目录；模型参数放 extra_params）
    GET  /v1/voices       列出可用音色
    POST /v1/voices       上传新音色（需 consent 授权声明）
    DELETE /v1/voices/{id} 删除音色

示例（curl）：
    curl -X POST http://127.0.0.1:8000/tts \
        -F "text=你好，欢迎使用 IndexTTS。" \
        -F "ref_audio=@examples/voice_01.wav" \
        -F "lang=ZH" \
        -o out.wav
"""

import argparse
import io
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

# 全局单例：模型 + GPU 推理锁
TTS = None
GPU_LOCK = threading.Lock()
SERVER_ARGS = None
# 音色目录（OpenAI 兼容接口的 voice 映射到这里的音频文件）
VOICES_DIR = os.environ.get("VOICES_DIR", "voices")


def build_tts(args):
    """构建 IndexTTS2 实例（与 webui.py 的 build_tts 逻辑一致）。"""
    from indextts.infer_v2_5 import IndexTTS2

    model_dir = args.model_dir
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")

    # 确保辅助模型（w2v-bert / campplus / bigvgan 等）已就位
    try:
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(model_dir)
    except Exception as e:  # 下载失败不阻塞启动，推理时若缺文件会再报错
        print(f">> 警告: ensure_models_available 失败: {e}")

    kwargs = dict(
        model_dir=model_dir,
        cfg_path=cfg_path,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
        use_accel=args.accel,
        use_torch_compile=args.torch_compile,
        use_qwen_emo=args.qwen_emo,
    )
    if args.bf16 and torch.cuda.is_bf16_supported():
        kwargs["use_bf16"] = True
    else:
        kwargs["use_bf16"] = False

    print(f">> 正在加载 IndexTTS2 模型 (model_dir={model_dir}) ...")
    t0 = time.time()
    tts = IndexTTS2(**kwargs)
    print(f">> 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时释放。"""
    global TTS
    TTS = build_tts(SERVER_ARGS)
    yield
    TTS = None


app = FastAPI(
    title="IndexTTS-2.5 API",
    description="IndexTTS-2.5 零样本语音合成 HTTP 服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许上传的音频扩展名
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _save_upload(upload: UploadFile) -> str:
    """把上传的音频保存到临时文件，返回路径。校验扩展名。"""
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext or '(无扩展名)'}，"
                   f"支持: {', '.join(sorted(ALLOWED_AUDIO_EXT))}",
        )
    suffix = ext or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


def _clean(path: Optional[str]):
    """删除临时文件（忽略错误）。"""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 音色管理（OpenAI 兼容 /v1/voices）
# ---------------------------------------------------------------------------

def _list_voices() -> list[str]:
    """列出音色目录中所有可用音色 id（文件名去扩展名）。"""
    if not os.path.isdir(VOICES_DIR):
        return []
    ids = []
    for name in sorted(os.listdir(VOICES_DIR)):
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_AUDIO_EXT:
            ids.append(os.path.splitext(name)[0])
    return ids


def _resolve_voice(voice: str) -> str:
    """把 voice id 解析成参考音频文件路径，找不到则 400。"""
    for ext in sorted(ALLOWED_AUDIO_EXT):
        path = os.path.join(VOICES_DIR, voice + ext)
        if os.path.isfile(path):
            return path
    raise HTTPException(
        status_code=400,
        detail=f"找不到音色 '{voice}'。可用音色: {', '.join(_list_voices()) or '(空，请先上传)'}",
    )


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """JSON 方式的 /tts 请求体（参考音频用本地路径）。"""
    text: str = Field(..., description="要合成的文本")
    ref_audio: str = Field(..., description="参考音频的本地路径（用于音色克隆）")
    lang: str = Field("ZH", description="语言: ZH / EN / JA / YUE 等")
    # 情感控制
    emo_ref_audio: Optional[str] = Field(None, description="情感参考音频路径（可选）")
    emo_alpha: float = Field(1.0, description="情感参考音频强度 0~1+")
    emo_vector: Optional[list] = Field(None, description="8 维情感向量 [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]")
    emo_text: Optional[str] = Field(None, description="情感描述文本（需 --qwen_emo）")
    # 采样参数
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 3
    temperature: float = 0.8
    num_beams: int = 1
    repetition_penalty: float = 1.0
    max_mel_tokens: int = 1500
    duration_factor: float = 1.0
    max_text_tokens_per_segment: int = 120


def _run_infer(
    text: str,
    ref_audio: str,
    lang: str,
    emo_ref_audio: Optional[str] = None,
    emo_alpha: float = 1.0,
    emo_vector: Optional[list] = None,
    emo_text: Optional[str] = None,
    do_sample: bool = True,
    top_p: float = 0.8,
    top_k: int = 3,
    temperature: float = 0.8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    max_mel_tokens: int = 1500,
    duration_factor: float = 1.0,
    max_text_tokens_per_segment: int = 120,
    stream: bool = False,
    use_emo_text: Optional[bool] = None,
    text_normalization: bool = True,
    use_random: bool = False,
):
    """调用 TTS.infer，返回 (sampling_rate, wav_int16_ndarray) 或生成器。"""
    if TTS is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    # use_emo_text 未显式指定时，沿用"有 emo_text 即启用"的旧行为
    if use_emo_text is None:
        use_emo_text = bool(emo_text)

    gen_kwargs = dict(
        do_sample=bool(do_sample),
        top_p=float(top_p),
        top_k=int(top_k) if int(top_k) > 0 else None,
        temperature=float(temperature),
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        max_mel_tokens=int(max_mel_tokens),
    )

    infer_kwargs = dict(
        spk_audio_prompt=ref_audio,
        text=text,
        output_path=None,  # 不写文件，直接拿 numpy 数据
        lang=lang,
        emo_audio_prompt=emo_ref_audio,
        emo_alpha=float(emo_alpha),
        emo_vector=emo_vector,
        use_emo_text=bool(use_emo_text),
        emo_text=emo_text,
        use_random=bool(use_random),
        text_normalization=bool(text_normalization),
        duration_factor=float(duration_factor),
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        stream_return=stream,
        **gen_kwargs,
    )

    with GPU_LOCK:  # 单卡串行，避免并发 OOM
        result = TTS.infer(**infer_kwargs)
    return result


def _result_to_wav_bytes(result) -> tuple[int, bytes]:
    """把 infer 的返回值转成 (sampling_rate, wav_bytes)。

    infer 返回 (sampling_rate, wav_data)，其中 wav_data 是 int16 numpy，
    形状 (N, 1)（已 .T 过），值域 [-32767, 32767]。这里复用项目自带的
    save_pcm_wav（与 webui 相同的 torchaudio 写盘路径）写到临时文件再读回，
    保证 WAV 格式兼容。
    """
    if result is None:
        raise HTTPException(status_code=500, detail="推理失败，未返回音频")
    if not (isinstance(result, tuple) and len(result) == 2):
        raise HTTPException(status_code=500, detail=f"意外的推理返回类型: {type(result)}")

    sampling_rate, wav_data = result
    # wav_data: int16 numpy (N, 1) -> torch (1, N)
    wav = torch.as_tensor(wav_data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.shape[0] != 1:
        wav = wav.t().contiguous()  # (N,1) -> (1,N)
    # save_pcm_wav 期望 PCM 尺度 ([-32767,32767]) 的 tensor
    if wav.dtype != torch.int16:
        wav = (wav * 32767).clamp(-32768, 32767).to(torch.int16)

    from indextts.utils.common import save_pcm_wav
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        save_pcm_wav(tmp.name, wav, int(sampling_rate))
        with open(tmp.name, "rb") as f:
            wav_bytes = f.read()
    finally:
        _clean(tmp.name)
    return int(sampling_rate), wav_bytes


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if TTS is not None else "loading",
        "model_loaded": TTS is not None,
        "device": getattr(TTS, "device", None),
        "gpu": torch.cuda.is_available(),
    }


@app.post("/tts")
async def tts(
    text: str = Form(..., description="要合成的文本"),
    ref_audio: UploadFile = File(..., description="参考音频（音色克隆）"),
    lang: str = Form("ZH", description="语言: ZH / EN / JA / YUE"),
    emo_ref_audio: Optional[UploadFile] = File(None, description="情感参考音频（可选）"),
    emo_alpha: float = Form(1.0, description="情感参考音频强度"),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(3),
    temperature: float = Form(0.8),
    num_beams: int = Form(1),
    repetition_penalty: float = Form(1.0),
    max_mel_tokens: int = Form(1500),
    duration_factor: float = Form(1.0),
    max_text_tokens_per_segment: int = Form(120),
):
    """multipart/form-data 方式合成，返回 WAV 二进制。"""
    ref_path = _save_upload(ref_audio)
    emo_path = _save_upload(emo_ref_audio) if emo_ref_audio is not None else None
    try:
        result = _run_infer(
            text=text, ref_audio=ref_path, lang=lang,
            emo_ref_audio=emo_path, emo_alpha=emo_alpha,
            do_sample=do_sample, top_p=top_p, top_k=top_k,
            temperature=temperature, num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens, duration_factor=duration_factor,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
        )
        sr, wav_bytes = _result_to_wav_bytes(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理出错: {e}")
    finally:
        _clean(ref_path)
        _clean(emo_path)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.post("/tts/json")
def tts_json(req: TTSRequest):
    """JSON 方式合成（参考音频用本地路径），返回 WAV 二进制。"""
    if not os.path.isfile(req.ref_audio):
        raise HTTPException(status_code=400, detail=f"参考音频不存在: {req.ref_audio}")
    try:
        result = _run_infer(
            text=req.text, ref_audio=req.ref_audio, lang=req.lang,
            emo_ref_audio=req.emo_ref_audio, emo_alpha=req.emo_alpha,
            emo_vector=req.emo_vector, emo_text=req.emo_text,
            do_sample=req.do_sample, top_p=req.top_p, top_k=req.top_k,
            temperature=req.temperature, num_beams=req.num_beams,
            repetition_penalty=req.repetition_penalty,
            max_mel_tokens=req.max_mel_tokens, duration_factor=req.duration_factor,
            max_text_tokens_per_segment=req.max_text_tokens_per_segment,
        )
        sr, wav_bytes = _result_to_wav_bytes(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理出错: {e}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="要合成的文本"),
    ref_audio: UploadFile = File(..., description="参考音频（音色克隆）"),
    lang: str = Form("ZH", description="语言: ZH / EN / JA / YUE"),
    emo_ref_audio: Optional[UploadFile] = File(None, description="情感参考音频（可选）"),
    emo_alpha: float = Form(1.0),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(3),
    temperature: float = Form(0.8),
    num_beams: int = Form(1),
    repetition_penalty: float = Form(1.0),
    max_mel_tokens: int = Form(1500),
    duration_factor: float = Form(1.0),
    max_text_tokens_per_segment: int = Form(120),
):
    """SSE 流式合成：边生成边推送 WAV 分块（base64），最后一条带元信息。"""
    ref_path = _save_upload(ref_audio)
    emo_path = _save_upload(emo_ref_audio) if emo_ref_audio is not None else None

    def event_gen():
        import base64
        try:
            gen = _run_infer(
                text=text, ref_audio=ref_path, lang=lang,
                emo_ref_audio=emo_path, emo_alpha=emo_alpha,
                do_sample=do_sample, top_p=top_p, top_k=top_k,
                temperature=temperature, num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens, duration_factor=duration_factor,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                stream=True,
            )
            for item in gen:
                if item is None:
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    sr, wav = item
                    wav = torch.as_tensor(wav)
                    if wav.dtype != torch.int16:
                        wav = (wav * 32767).clamp(-32768, 32767).to(torch.int16)
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    b64 = base64.b64encode(wav.cpu().numpy().tobytes()).decode()
                    yield f"data: {b64}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            _clean(ref_path)
            _clean(emo_path)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# OpenAI 兼容接口（/v1/audio/speech + /v1/voices）
# ---------------------------------------------------------------------------

class OpenAISpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 请求体（额外字段透传给 IndexTTS）。

    模型特有参数（lang / 情感 / 文本规范化等）推荐放进 extra_params 对象，
    与 vLLM-Omni 的 OpenAI 兼容约定一致；也兼容直接平铺在顶层的旧写法。
    """
    model: str = Field("indextts", description="模型名（忽略，兼容用）")
    input: str = Field(..., description="要合成的文本")
    voice: str = Field(..., description="音色 id（对应 voices 目录中的音频文件名）")
    response_format: str = Field("wav", description="响应格式（本服务始终返回 16-bit PCM WAV）")
    speed: Optional[float] = Field(None, description="语速 [0.5, 2.0]，>1 更快；映射到 duration_factor=1/speed")
    # 模型特有参数（推荐放这里，优先级高于下方平铺字段）
    extra_params: Optional[dict] = Field(
        None,
        description="模型特有参数: lang / text_normalization / emo_vector / emo_alpha / emo_text / use_emo_text / emo_audio / use_random",
    )
    # IndexTTS 特有参数（平铺写法，兼容用；extra_params 中的同名键会覆盖）
    lang: str = Field("ZH", description="语言: ZH / EN / JA / YUE")
    emo_alpha: float = Field(1.0, description="情感强度")
    emo_vector: Optional[list] = Field(None, description="8 维情感向量 [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]")
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 3
    temperature: float = 0.8
    num_beams: int = 1
    repetition_penalty: float = 1.0
    max_mel_tokens: int = 1500
    max_text_tokens_per_segment: int = 120


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    """OpenAI 兼容的语音合成接口，返回 16-bit PCM WAV。

    任何 OpenAI SDK 客户端只需改 base_url 即可使用：
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        client.audio.speech.create(model="indextts", voice="voice_01", input="你好")
    """
    # speed 范围校验（对应 duration_factor，越界 422）
    speed = req.speed if req.speed is not None else 1.0
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=422, detail="speed 必须在 [0.5, 2.0] 范围内")
    duration_factor = 1.0 / speed

    # 合并 extra_params（模型特有参数，优先级高于顶层平铺字段）
    extra = req.extra_params or {}
    lang = extra.get("lang", req.lang)
    emo_alpha = extra.get("emo_alpha", req.emo_alpha)
    emo_vector = extra.get("emo_vector", req.emo_vector)
    emo_text = extra.get("emo_text", None)
    use_emo_text = extra.get("use_emo_text", None)
    emo_audio = extra.get("emo_audio", None)
    use_random = bool(extra.get("use_random", False))
    text_normalization = bool(extra.get("text_normalization", True))

    ref_path = _resolve_voice(req.voice)
    try:
        result = _run_infer(
            text=req.input, ref_audio=ref_path, lang=lang,
            emo_ref_audio=emo_audio, emo_alpha=emo_alpha, emo_vector=emo_vector,
            emo_text=emo_text, use_emo_text=use_emo_text,
            do_sample=req.do_sample, top_p=req.top_p, top_k=req.top_k,
            temperature=req.temperature, num_beams=req.num_beams,
            repetition_penalty=req.repetition_penalty,
            max_mel_tokens=req.max_mel_tokens, duration_factor=duration_factor,
            max_text_tokens_per_segment=req.max_text_tokens_per_segment,
            text_normalization=text_normalization, use_random=use_random,
        )
        sr, wav_bytes = _result_to_wav_bytes(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理出错: {e}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.get("/v1/voices")
def list_voices():
    """列出可用音色。"""
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in _list_voices()],
    }


@app.post("/v1/voices")
async def create_voice(
    file: UploadFile = File(..., description="参考音频（3-10 秒清晰人声效果最佳）"),
    name: Optional[str] = Form(None, description="音色 id（默认用文件名）"),
    consent: str = Form(..., description="授权声明（必填：克隆真人声音需获得本人同意）"),
    speaker_description: Optional[str] = Form(None, description="音色描述（可选）"),
):
    """上传新音色，保存到 voices 目录（volume 持久化）。

    借鉴 vLLM-Omni 的音色存储约定：上传必须携带 consent 授权声明。
    """
    if not consent or not consent.strip():
        raise HTTPException(status_code=400, detail="consent 字段必填（音色克隆授权声明）")
    os.makedirs(VOICES_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext or '(无扩展名)'}，"
                   f"支持: {', '.join(sorted(ALLOWED_AUDIO_EXT))}",
        )
    voice_id = (name or os.path.splitext(file.filename or "")[0]).strip()
    if not voice_id or not all(c.isalnum() or c in "-_" for c in voice_id):
        raise HTTPException(status_code=400, detail=f"非法音色 id: '{voice_id}'（只允许字母/数字/-/_）")
    dest = os.path.join(VOICES_DIR, voice_id + ext)
    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"音色 '{voice_id}' 已存在，请先 DELETE 再上传")
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return {"object": "voice", "id": voice_id, "filename": os.path.basename(dest)}


@app.delete("/v1/voices/{voice_id}")
def delete_voice(voice_id: str):
    """删除音色。"""
    path = _resolve_voice(voice_id)
    os.remove(path)
    return {"object": "voice.deleted", "id": voice_id, "deleted": True}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    global SERVER_ARGS
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 FastAPI 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # 支持 MODEL_DIR 环境变量（Docker 部署时挂载模型目录用）
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("MODEL_DIR", "checkpoints"))
    # 音色目录（OpenAI 兼容接口用），支持 VOICES_DIR 环境变量
    parser.add_argument("--voices_dir", type=str,
                        default=os.environ.get("VOICES_DIR", "voices"))
    parser.add_argument("--bf16", action="store_true", help="使用 bfloat16（需 GPU 支持）")
    parser.add_argument("--qwen_emo", action="store_true", help="加载 QwenEmotion（支持情感文本控制）")
    parser.add_argument("--accel", action="store_true", help="启用 GPT2 加速引擎（需 flash-attn）")
    parser.add_argument("--deepspeed", action="store_true", help="启用 DeepSpeed")
    parser.add_argument("--torch_compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--cuda_kernel", action="store_true", help="启用 BigVGAN 自定义 CUDA kernel")
    args = parser.parse_args()
    SERVER_ARGS = args
    global VOICES_DIR
    VOICES_DIR = args.voices_dir
    os.makedirs(VOICES_DIR, exist_ok=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

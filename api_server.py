"""IndexTTS-2.5 FastAPI HTTP service.

Exposes IndexTTS2 inference as a REST API for HTTP-based integration.

Startup (note: local service must unset proxy, otherwise gradio/uvicorn self-check will fail through proxy):

    cd /home/iamyb/github/index-tts
    env -u http_proxy -u https_proxy -u all_proxy \
        USE_MODELSCOPE=false HF_ENDPOINT=https://hf-mirror.com \
        /home/iamyb/miniconda3/envs/indextts/bin/python api_server.py --port 8000

Endpoints:
    GET  /health          Health check
    POST /tts             Text + reference audio -> WAV audio
    POST /tts/json        JSON-based synthesis (reference audio via local path)
    POST /tts/stream      Same as above, but returns audio chunks via SSE streaming (low first-packet latency)
    POST /v1/audio/speech OpenAI-compatible synthesis (voice maps to voices directory; model params in extra_params)
    GET  /v1/voices       List available voices
    POST /v1/voices       Upload a new voice (requires consent declaration)
    DELETE /v1/voices/{id} Delete a voice

Example (curl):
    curl -X POST http://127.0.0.1:8000/tts \
        -F "text=Hello, welcome to IndexTTS." \
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
# Model loading
# ---------------------------------------------------------------------------

# Global singleton: model + GPU inference lock
TTS = None
GPU_LOCK = threading.Lock()
SERVER_ARGS = None
# Voices directory (OpenAI-compatible voice IDs map to audio files here)
VOICES_DIR = os.environ.get("VOICES_DIR", "voices")


def build_tts(args):
    """Build an IndexTTS2 instance (same logic as build_tts in webui.py)."""
    from indextts.infer_v2_5 import IndexTTS2

    model_dir = args.model_dir
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Ensure auxiliary models (w2v-bert / campplus / bigvgan, etc.) are in place
    try:
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(model_dir)
    except Exception as e:  # Download failure does not block startup; missing files will error at inference time
        print(f">> Warning: ensure_models_available failed: {e}")

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

    print(f">> Loading IndexTTS2 model (model_dir={model_dir}) ...")
    t0 = time.time()
    tts = IndexTTS2(**kwargs)
    print(f">> Model loaded in {time.time() - t0:.1f}s")
    return tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: load model on startup, release on shutdown."""
    global TTS
    TTS = build_tts(SERVER_ARGS)
    yield
    TTS = None


app = FastAPI(
    title="IndexTTS-2.5 API",
    description="IndexTTS-2.5 zero-shot text-to-speech HTTP service",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed audio extensions for upload
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _save_upload(upload: UploadFile) -> str:
    """Save uploaded audio to a temp file and return the path. Validates extension."""
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext or '(no extension)'}."
                   f" Supported: {', '.join(sorted(ALLOWED_AUDIO_EXT))}",
        )
    suffix = ext or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


def _clean(path: Optional[str]):
    """Remove a temp file (ignore errors)."""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Voice management (OpenAI-compatible /v1/voices)
# ---------------------------------------------------------------------------

def _list_voices() -> list[str]:
    """List all available voice IDs in the voices directory (filename without extension)."""
    if not os.path.isdir(VOICES_DIR):
        return []
    ids = []
    for name in sorted(os.listdir(VOICES_DIR)):
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_AUDIO_EXT:
            ids.append(os.path.splitext(name)[0])
    return ids


def _resolve_voice(voice: str) -> str:
    """Resolve a voice ID to a reference audio file path; raises 400 if not found."""
    for ext in sorted(ALLOWED_AUDIO_EXT):
        path = os.path.join(VOICES_DIR, voice + ext)
        if os.path.isfile(path):
            return path
    raise HTTPException(
        status_code=400,
        detail=f"Voice '{voice}' not found. Available voices: {', '.join(_list_voices()) or '(empty, please upload first)'}",
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """JSON request body for /tts (reference audio via local path)."""
    text: str = Field(..., description="Text to synthesize")
    ref_audio: str = Field(..., description="Local path to reference audio (for voice cloning)")
    lang: str = Field("ZH", description="Language: ZH / EN / JA / YUE, etc.")
    # Emotion control
    emo_ref_audio: Optional[str] = Field(None, description="Emotion reference audio path (optional)")
    emo_alpha: float = Field(1.0, description="Emotion reference audio intensity 0~1+")
    emo_vector: Optional[list] = Field(None, description="8-dim emotion vector [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]")
    emo_text: Optional[str] = Field(None, description="Emotion description text (requires --qwen_emo)")
    # Sampling parameters
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
    """Call TTS.infer, returning (sampling_rate, wav_int16_ndarray) or a generator."""
    if TTS is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    # When use_emo_text is not explicitly set, fall back to legacy behavior: enable if emo_text is present
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
        output_path=None,  # Don't write to file; get numpy data directly
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

    with GPU_LOCK:  # Single-GPU serialization to avoid concurrent OOM
        result = TTS.infer(**infer_kwargs)
    return result


def _result_to_wav_bytes(result) -> tuple[int, bytes]:
    """Convert infer's return value to (sampling_rate, wav_bytes).

    infer returns (sampling_rate, wav_data), where wav_data is int16 numpy,
    shape (N, 1) (already transposed), range [-32767, 32767]. This reuses the
    project's save_pcm_wav (same torchaudio write path as webui) to write to a
    temp file and read it back, ensuring WAV format compatibility.
    """
    if result is None:
        raise HTTPException(status_code=500, detail="Inference failed, no audio returned")
    if not (isinstance(result, tuple) and len(result) == 2):
        raise HTTPException(status_code=500, detail=f"Unexpected inference return type: {type(result)}")

    sampling_rate, wav_data = result
    # wav_data: int16 numpy (N, 1) -> torch (1, N)
    wav = torch.as_tensor(wav_data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.shape[0] != 1:
        wav = wav.t().contiguous()  # (N,1) -> (1,N)
    # save_pcm_wav expects a tensor in PCM scale ([-32767, 32767])
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
# Routes
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
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio (voice cloning)"),
    lang: str = Form("ZH", description="Language: ZH / EN / JA / YUE"),
    emo_ref_audio: Optional[UploadFile] = File(None, description="Emotion reference audio (optional)"),
    emo_alpha: float = Form(1.0, description="Emotion reference audio intensity"),
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
    """Synthesize via multipart/form-data, returns WAV binary."""
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
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
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
    """Synthesize via JSON (reference audio via local path), returns WAV binary."""
    if not os.path.isfile(req.ref_audio):
        raise HTTPException(status_code=400, detail=f"Reference audio not found: {req.ref_audio}")
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
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio (voice cloning)"),
    lang: str = Form("ZH", description="Language: ZH / EN / JA / YUE"),
    emo_ref_audio: Optional[UploadFile] = File(None, description="Emotion reference audio (optional)"),
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
    """SSE streaming synthesis: pushes WAV chunks (base64) as they are generated; last event carries metadata."""
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
# OpenAI-compatible endpoints (/v1/audio/speech + /v1/voices)
# ---------------------------------------------------------------------------

class OpenAISpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request body (extra fields are passed through to IndexTTS).

    Model-specific parameters (lang / emotion / text normalization, etc.) are recommended
    to be placed in the extra_params object, consistent with vLLM-Omni's OpenAI-compatible
    convention; flat top-level fields are also supported for backward compatibility.
    """
    model: str = Field("indextts", description="Model name (ignored, for compatibility)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice ID (maps to an audio filename in the voices directory)")
    response_format: str = Field("wav", description="Response format (this service always returns 16-bit PCM WAV)")
    speed: Optional[float] = Field(None, description="Speed [0.5, 2.0], >1 is faster; maps to duration_factor=1/speed")
    # Model-specific parameters (recommended here; takes priority over flat fields below)
    extra_params: Optional[dict] = Field(
        None,
        description="Model-specific params: lang / text_normalization / emo_vector / emo_alpha / emo_text / use_emo_text / emo_audio / use_random",
    )
    # IndexTTS-specific parameters (flat style, for compatibility; same-named keys in extra_params override)
    lang: str = Field("ZH", description="Language: ZH / EN / JA / YUE")
    emo_alpha: float = Field(1.0, description="Emotion intensity")
    emo_vector: Optional[list] = Field(None, description="8-dim emotion vector [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]")
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
    """OpenAI-compatible speech synthesis endpoint, returns 16-bit PCM WAV.

    Any OpenAI SDK client can use it by just changing base_url:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        client.audio.speech.create(model="indextts", voice="voice_01", input="Hello")
    """
    # Speed range validation (maps to duration_factor; out of range returns 422)
    speed = req.speed if req.speed is not None else 1.0
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=422, detail="speed must be in the range [0.5, 2.0]")
    duration_factor = 1.0 / speed

    # Merge extra_params (model-specific params, takes priority over flat top-level fields)
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
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.get("/v1/voices")
def list_voices():
    """List available voices."""
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in _list_voices()],
    }


@app.post("/v1/voices")
async def create_voice(
    file: UploadFile = File(..., description="Reference audio (3-10 seconds of clear speech works best)"),
    name: Optional[str] = Form(None, description="Voice ID (defaults to filename)"),
    consent: str = Form(..., description="Consent declaration (required: cloning a real person's voice requires their consent)"),
    speaker_description: Optional[str] = Form(None, description="Voice description (optional)"),
):
    """Upload a new voice, saved to the voices directory (volume-persisted).

    Follows vLLM-Omni's voice storage convention: upload must include a consent declaration.
    """
    if not consent or not consent.strip():
        raise HTTPException(status_code=400, detail="consent field is required (voice cloning consent declaration)")
    os.makedirs(VOICES_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext or '(no extension)'}"
                   f". Supported: {', '.join(sorted(ALLOWED_AUDIO_EXT))}",
        )
    voice_id = (name or os.path.splitext(file.filename or "")[0]).strip()
    if not voice_id or not all(c.isalnum() or c in "-_" for c in voice_id):
        raise HTTPException(status_code=400, detail=f"Invalid voice ID: '{voice_id}' (only alphanumeric, -, _ allowed)")
    dest = os.path.join(VOICES_DIR, voice_id + ext)
    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"Voice '{voice_id}' already exists; DELETE it first before uploading")
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return {"object": "voice", "id": voice_id, "filename": os.path.basename(dest)}


@app.delete("/v1/voices/{voice_id}")
def delete_voice(voice_id: str):
    """Delete a voice."""
    path = _resolve_voice(voice_id)
    os.remove(path)
    return {"object": "voice.deleted", "id": voice_id, "deleted": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global SERVER_ARGS
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 FastAPI service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # Supports MODEL_DIR env var (for mounting model directory in Docker deployment)
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("MODEL_DIR", "checkpoints"))
    # Voices directory (for OpenAI-compatible endpoints), supports VOICES_DIR env var
    parser.add_argument("--voices_dir", type=str,
                        default=os.environ.get("VOICES_DIR", "voices"))
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (requires GPU support)")
    parser.add_argument("--qwen_emo", action="store_true", help="Load QwenEmotion (enables emotion text control)")
    parser.add_argument("--accel", action="store_true", help="Enable GPT2 acceleration engine (requires flash-attn)")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--cuda_kernel", action="store_true", help="Enable BigVGAN custom CUDA kernel")
    args = parser.parse_args()
    SERVER_ARGS = args
    global VOICES_DIR
    VOICES_DIR = args.voices_dir
    os.makedirs(VOICES_DIR, exist_ok=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

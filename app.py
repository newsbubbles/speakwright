"""
Speakwright — a local voice sidecar: STT (faster-whisper) + TTS (Kokoro-82M),
with an optional LLM pass that rewrites text to sound natural read aloud before
it's spoken. Exposed over the OpenAI audio API so any OpenAI-compatible client
can use it.

Bound to 127.0.0.1 only. No API key required.

Endpoints (OpenAI-compatible):
  POST /v1/audio/transcriptions   (multipart 'file' [+ 'model'])     -> { text }
  POST /v1/audio/speech           ({ input, voice?, response_format? }) -> audio bytes
Also keeps the legacy /transcribe and /tts for backward compat.
"""
import asyncio
import io
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-sidecar")

PORT = int(os.getenv("VOICE_PORT", "8765"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").lower()
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")
KOKORO_SR = 24000

# OpenAI's stock voice names — if a client sends one, fall back to our Kokoro default.
OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "coral", "sage"}

# Voice-readiness agent (optional): an LLM that rewrites a message to sound natural spoken,
# replacing the regex when configured. Any OpenAI-compatible chat endpoint.
VOICE_AGENT_MODEL = os.getenv("VOICE_AGENT_MODEL", "")
VOICE_AGENT_BASE_URL = os.getenv("VOICE_AGENT_BASE_URL", "https://openrouter.ai/api/v1")
VOICE_AGENT_API_KEY = os.getenv("VOICE_AGENT_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")

_whisper = None
_whisper_lock = asyncio.Lock()
_kpipe = None
_kpipe_lock = asyncio.Lock()


async def get_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    async with _whisper_lock:
        if _whisper is not None:
            return _whisper

        def _load():
            from faster_whisper import WhisperModel
            if WHISPER_DEVICE == "cuda":
                try:
                    return WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                except Exception as e:  # noqa: BLE001
                    logger.warning("[WHISPER] CUDA failed (%s), CPU fallback", e)
            return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

        _whisper = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("[WHISPER] ready")
        return _whisper


async def get_kokoro():
    global _kpipe
    if _kpipe is not None:
        return _kpipe
    async with _kpipe_lock:
        if _kpipe is not None:
            return _kpipe

        def _load():
            import torch
            from kokoro import KPipeline
            device = os.getenv("KOKORO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("[KOKORO] loading on %s", device)
            return KPipeline(lang_code=KOKORO_LANG, device=device)

        _kpipe = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("[KOKORO] ready")
        return _kpipe


def prepare_text_for_tts(text: str) -> str:
    """Strip/transform markdown so it sounds natural when spoken."""
    text = re.sub(r"```[\s\S]*?```", " code block ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


VOICE_AGENT_SYSTEM = (
    "You rewrite a chat assistant's message so it sounds natural read ALOUD, not on screen. "
    "Keep the meaning and the facts; do not add commentary or new information.\n"
    "- Say symbols as words in context: -> as 'to' or 'becomes', & 'and', @ 'at', % 'percent', "
    "/ 'slash' or 'or' as it fits, == 'equals'.\n"
    "- Do not read code or data verbatim. For a code block, give a one-line spoken summary of what "
    "it does, or say you are skipping it. For inline code like foo(), say it naturally.\n"
    "- Turn structure into speech: bullet and numbered lists become flowing sentences with 'first', "
    "'next', 'also'; a table becomes a sentence or two of summary, not a cell-by-cell read; "
    "headings become a spoken transition or are dropped.\n"
    "- Paths and URLs: say the meaningful part ('the app.ts file', 'a link to example dot com'), "
    "not every slash.\n"
    "- Drop markdown emphasis and emoji unless an emoji carries real meaning.\n"
    "- Be a little more concise where the original was visually dense.\n"
    "Output only the spoken text. No markdown."
)

_llm = None


def _agent_enabled() -> bool:
    return bool(VOICE_AGENT_MODEL and VOICE_AGENT_API_KEY)


def _get_llm():
    global _llm
    if _llm is None:
        from openai import AsyncOpenAI
        _llm = AsyncOpenAI(base_url=VOICE_AGENT_BASE_URL, api_key=VOICE_AGENT_API_KEY)
    return _llm


async def voice_ready_text(text: str) -> str:
    """Make text sound natural spoken. Uses the LLM agent if configured, else the regex baseline."""
    baseline = prepare_text_for_tts(text)
    if not _agent_enabled():
        return baseline
    try:
        client = _get_llm()
        resp = await client.chat.completions.create(
            model=VOICE_AGENT_MODEL,
            messages=[
                {"role": "system", "content": VOICE_AGENT_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            logger.info("[VOICE-AGENT] rewrote %d -> %d chars", len(text), len(out))
            return out
        return baseline
    except Exception as e:  # noqa: BLE001
        logger.warning("[VOICE-AGENT] rewrite failed, using regex baseline: %s", e)
        return baseline


async def _transcribe_bytes(content: bytes, filename: str) -> str:
    suffix = Path(filename or "rec.webm").suffix or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        model = await get_whisper()

        def _run():
            segments, _info = model.transcribe(tmp_path, beam_size=5)
            return "".join(seg.text for seg in segments).strip()

        return await asyncio.get_event_loop().run_in_executor(None, _run)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _synth_wav(text: str, voice: Optional[str]) -> bytes:
    clean = await voice_ready_text(text)
    chosen = voice or KOKORO_VOICE
    if not chosen or chosen in OPENAI_VOICES:
        chosen = KOKORO_VOICE
    pipeline = await get_kokoro()

    def _synth():
        chunks = [audio for _gs, _ps, audio in pipeline(clean, voice=chosen)]
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        full = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        buf = io.BytesIO()
        sf.write(buf, full, KOKORO_SR, format="WAV")
        return buf.getvalue()

    return await asyncio.get_event_loop().run_in_executor(None, _synth)


app = FastAPI(title="Speakwright voice sidecar")


@app.get("/health")
async def health():
    return {"status": "ok", "whisper_loaded": _whisper is not None, "kokoro_loaded": _kpipe is not None}


# ---- OpenAI-compatible endpoints ----------------------------------------------
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form(None)):
    start = time.time()
    content = await file.read()
    try:
        text = await _transcribe_bytes(content, file.filename or "rec.webm")
        logger.info("[STT] %dms: %s", int((time.time() - start) * 1000), text[:80])
        return {"text": text}
    except Exception as e:  # noqa: BLE001
        logger.error("[STT] failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input cannot be empty")
    start = time.time()
    try:
        wav = await _synth_wav(req.input, req.voice)
        logger.info("[TTS] %d bytes in %dms", len(wav), int((time.time() - start) * 1000))
        return Response(content=wav, media_type="audio/wav")
    except Exception as e:  # noqa: BLE001
        logger.error("[TTS] failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ---- Legacy endpoints (kept for backward compatibility) -----------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    content = await audio.read()
    try:
        text = await _transcribe_bytes(content, audio.filename or "rec.webm")
        return {"text": text}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/tts")
async def tts(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        wav = await _synth_wav(text, None)
        return Response(content=wav, media_type="audio/wav")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")

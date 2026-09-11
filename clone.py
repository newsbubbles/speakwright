"""
clone.py: the cloned voice, as its own sidecar.

Chatterbox-Turbo pins an older torch than the rest of speakwright, so it
lives in .venv-clone and answers the same speech endpoint on its own port.
app.py routes any voice that has a reference clip in voices/ here and
falls back to Kokoro when this process is down or the GPU is busy with a
render, which on an 8 GB card is most of a run.

    .venv-clone/Scripts/python clone.py            # 127.0.0.1:8766

    POST /v1/audio/speech  { input, voice? }  -> audio/wav (24 kHz)
    GET  /health
"""
import asyncio
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

# the weights land next to the sidecar unless the caller says otherwise:
# the shared Hugging Face cache sits on the small system drive
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".hf"))

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-clone")

PORT = int(os.getenv("CLONE_PORT", "8766"))
VOICES = Path(os.getenv("CLONE_VOICES", str(Path(__file__).parent / "voices")))
DEFAULT = os.getenv("CLONE_DEFAULT", "nate")
DEVICE = os.getenv("CLONE_DEVICE", "cuda")
# below this much free VRAM a render is in progress: refuse, let Kokoro talk
MIN_FREE_GB = float(os.getenv("CLONE_MIN_FREE_GB", "1.2"))

_model = None
_lock = asyncio.Lock()
_cond_clip = None      # the clip the model is currently conditioned on
_gen_lock = asyncio.Lock()   # one generation at a time: the model keeps state


def clip_for(voice: Optional[str]) -> Path:
    name = (voice or DEFAULT).strip()
    p = VOICES / f"{name}.wav"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no reference clip voices/{name}.wav")
    return p


def free_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


async def get_model():
    global _model
    if _model is not None:
        return _model
    async with _lock:
        if _model is not None:
            return _model

        def _load():
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            logger.info("[CLONE] loading Chatterbox-Turbo on %s", DEVICE)
            return ChatterboxTurboTTS.from_pretrained(device=DEVICE)

        _model = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("[CLONE] ready (sr %s)", getattr(_model, "sr", "?"))
        return _model


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"


app = FastAPI(title="Speakwright voice clone")


@app.get("/health")
async def health():
    return {"status": "ok", "loaded": _model is not None, "free_gb": round(free_gb(), 2),
            "voices": sorted(p.stem for p in VOICES.glob("*.wav"))}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input cannot be empty")
    clip = clip_for(req.voice)
    if DEVICE == "cuda" and _model is None and free_gb() < MIN_FREE_GB + 1.5:
        raise HTTPException(status_code=503, detail=f"gpu busy ({free_gb():.1f} GB free): use the stock voice")
    if DEVICE == "cuda" and _model is not None and free_gb() < MIN_FREE_GB:
        raise HTTPException(status_code=503, detail=f"gpu busy ({free_gb():.1f} GB free): use the stock voice")
    start = time.time()
    model = await get_model()

    def _synth():
        global _cond_clip
        import torch
        with torch.inference_mode():
            # conditioning on the clip costs seconds; do it once per voice
            if _cond_clip != clip:
                model.prepare_conditionals(str(clip))
                _cond_clip = clip
            wav = model.generate(text)
        audio = wav.squeeze().detach().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio, int(model.sr), format="WAV")
        return buf.getvalue()

    try:
        async with _gen_lock:
            out = await asyncio.get_event_loop().run_in_executor(None, _synth)
    except Exception as e:  # noqa: BLE001
        logger.error("[CLONE] failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"clone failed: {e}")
    logger.info("[CLONE] %s: %d bytes in %dms", clip.stem, len(out), int((time.time() - start) * 1000))
    return Response(content=out, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")

"""
Speakwright — a local voice sidecar: STT (faster-whisper) + TTS (Kokoro-82M),
with an optional LLM pass that rewrites text to sound natural read aloud before
it's spoken. Exposed over the OpenAI audio API so any OpenAI-compatible client
can use it.

Bound to 127.0.0.1 only. No API key required.

Endpoints (OpenAI-compatible):
  POST /v1/audio/transcriptions   (multipart 'file' [+ 'model'])     -> { text }
       + response_format=verbose_json -> { text, duration, words: [{word, start, end, probability}] }
       + model=parakeet -> Parakeet TDT 0.6B v2 on the CPU instead of whisper (STT_ENGINE sets the default)
  POST /v1/audio/speech           ({ input, voice?, response_format? }) -> audio bytes
Also keeps the legacy /transcribe and /tts for backward compat.
"""
import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-sidecar")

PORT = int(os.getenv("VOICE_PORT", "8765"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").lower()
WHISPER_LANG = os.getenv("WHISPER_LANG", "en")   # empty = detect
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "")   # force a CTranslate2 compute type
# A second speech-to-text engine for files (voice notes): NVIDIA Parakeet TDT 0.6B v2 through onnx-asr (CC-BY-4.0),
# int8 on the CPU, so it never takes the GPU. A request picks it with model="parakeet"; STT_ENGINE sets the default
# (whisper). The ear (live wake-word speech) always uses whisper. On a 4-core i7-6700HQ with 2 threads it reads
# 3.9 s of speech in about 1.2 s; whisper small.en on the GPU takes about 0.8 s.
STT_ENGINE = os.getenv("STT_ENGINE", "whisper").lower()
PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v2")
PARAKEET_THREADS = int(os.getenv("PARAKEET_THREADS", "2"))   # the live music engine shares these cores

# The ear: a hot mic stream with a wake word and VAD end-pointing. Time is
# audio time (frames counted), never the wall clock, so a test can stream
# faster than real time and behave the same.
EAR_SR = 16000
EAR_FRAME = 1280                                        # 80 ms, openWakeWord's frame
WAKE_MODEL = os.getenv("WAKE_MODEL", "hey_jarvis")
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
END_SILENCE_S = float(os.getenv("END_SILENCE_S", "0.7"))
WAKE_PATIENCE_S = 3.0                                   # wake word, then nothing: forget it
MAX_UTTERANCE_S = 20.0
PREROLL_FRAMES = 6                                      # ~0.5 s kept before speech is noticed
NO_SPEECH_MAX = float(os.getenv("NO_SPEECH_MAX", "0.6"))
LOGPROB_MIN = float(os.getenv("LOGPROB_MIN", "-1.0"))
WAKE_STRIP = re.compile(r"^\W*(hey|hi|ok|okay)?\W*jarvis\W*", re.I)
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
# Cloned voices live in clone.py (its own venv, port 8766): any voice with a
# reference clip in voices/ goes there, Kokoro answers when it cannot.
CLONE_URL = os.getenv("VOICE_CLONE_URL", "http://127.0.0.1:8766")
VOICES_DIR = Path(__file__).parent / "voices"
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", KOKORO_VOICE)
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
_parakeet = None
_parakeet_lock = asyncio.Lock()

# Idle unload: after VOICE_IDLE_UNLOAD_S with no use and no ear stream open, whisper drops its GPU weights (about 2 s
# to come back, from the OS file cache) and Kokoro moves to CPU RAM (about 0.5 s back). An open ear stream (the VR
# stage) pins both. Measured on a GTX 1080 (10-22): about 0.6 GB of VRAM and 0.3 GB of commit come back; the two
# CUDA contexts (about 3 GB of commit) stay while the process lives. 0 turns it off.
IDLE_UNLOAD_S = float(os.getenv("VOICE_IDLE_UNLOAD_S", "600"))
_state = {"whisper_compute": None, "whisper_dev": None, "kokoro_dev": None, "kokoro_parked": False,
          "last_use": time.time(), "busy": 0, "ears": 0, "unloads": 0, "resumes": 0}
_idle_lock = asyncio.Lock()


def _resume(kind):
    """Bring a parked model back to the GPU (in a worker thread)."""
    if kind == "whisper" and _whisper is not None and _state["whisper_dev"] == "cuda" \
            and not _whisper.model.model_is_loaded:
        t0 = time.time()
        _whisper.model.load_model()
        _state["resumes"] += 1
        logger.info("[IDLE] whisper back on the GPU in %dms", int((time.time() - t0) * 1000))
    if kind == "kokoro" and _kpipe is not None and _state["kokoro_parked"]:
        t0 = time.time()
        _kpipe.model.to("cuda")
        _state["kokoro_parked"] = False
        _state["resumes"] += 1
        logger.info("[IDLE] kokoro back on the GPU in %dms", int((time.time() - t0) * 1000))


def _unload():
    global _parakeet
    did = []
    if _parakeet is not None:                          # CPU RAM only; reloads from disk in a few seconds
        _parakeet = None
        did.append("parakeet")
    if _whisper is not None and _state["whisper_dev"] == "cuda" and _whisper.model.model_is_loaded:
        _whisper.model.unload_model()
        did.append("whisper")
    if _kpipe is not None and _state["kokoro_dev"] == "cuda" and not _state["kokoro_parked"]:
        import torch
        _kpipe.model.to("cpu")
        torch.cuda.empty_cache()
        _state["kokoro_parked"] = True
        did.append("kokoro")
    return did


class _Use:
    """A model in use: brought back if idle parked it, and the idle timer held off until the work is done."""

    def __init__(self, kind):
        self.kind = kind

    async def __aenter__(self):
        async with _idle_lock:
            _state["busy"] += 1
            _state["last_use"] = time.time()
            try:
                await asyncio.get_event_loop().run_in_executor(None, _resume, self.kind)
            except Exception:
                _state["busy"] -= 1
                raise

    async def __aexit__(self, *exc):
        _state["busy"] -= 1
        _state["last_use"] = time.time()


async def _idle_watch():
    while IDLE_UNLOAD_S > 0:
        await asyncio.sleep(min(30.0, IDLE_UNLOAD_S))
        if _state["busy"] or _state["ears"] or time.time() - _state["last_use"] < IDLE_UNLOAD_S:
            continue
        async with _idle_lock:
            if _state["busy"] or _state["ears"]:
                continue
            try:
                did = await asyncio.get_event_loop().run_in_executor(None, _unload)
            except Exception as e:  # noqa: BLE001
                logger.warning("[IDLE] unload failed: %s", e)
                continue
        if did:
            _state["unloads"] += 1
            logger.info("[IDLE] %ds unused: parked %s", int(time.time() - _state["last_use"]), ", ".join(did))


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
                # Pascal cards (GTX 10xx) have no float16 in CTranslate2, but they do run int8_float32: on a
                # GTX 1080, small.en takes 0.3 GB of VRAM and of commit that way, against 1.1 GB each in
                # float32, with the same words. float32 on the GPU is the last resort, still ahead of the CPU.
                try:
                    import ctranslate2
                    have = ctranslate2.get_supported_compute_types("cuda")
                except Exception:  # noqa: BLE001
                    have = {"float16", "int8_float32", "float32"}
                order = [WHISPER_COMPUTE] + [c for c in ("float16", "int8_float32", "float32") if c in have]
                for ct in [c for c in dict.fromkeys(order) if c]:
                    try:
                        m = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type=ct)
                        logger.info("[WHISPER] cuda %s", ct)
                        _state["whisper_compute"], _state["whisper_dev"] = ct, "cuda"
                        return m
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[WHISPER] cuda %s failed (%s)", ct, e)
                logger.warning("[WHISPER] CPU fallback")
            _state["whisper_compute"], _state["whisper_dev"] = "int8", "cpu"
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
            _state["kokoro_dev"] = device
            return KPipeline(lang_code=KOKORO_LANG, device=device)

        _kpipe = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("[KOKORO] ready")
        return _kpipe


async def get_parakeet():
    global _parakeet
    if _parakeet is not None:
        return _parakeet
    async with _parakeet_lock:
        if _parakeet is not None:
            return _parakeet

        def _load():
            try:
                import onnx_asr
                import onnxruntime as ort
            except ImportError:
                raise HTTPException(status_code=400, detail="the parakeet engine is not installed here: "
                                    "uv pip install --python .venv/Scripts/python.exe --no-deps onnx-asr "
                                    "(the int8 model, about 631 MB, downloads on first use); or use model=whisper")
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = PARAKEET_THREADS, 1
            return onnx_asr.load_model(PARAKEET_MODEL, quantization="int8", sess_options=so).with_timestamps()

        t0 = time.time()
        _parakeet = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("[PARAKEET] ready in %ds", int(time.time() - t0))
        return _parakeet


def _parakeet_words(r, duration):
    """Word times from Parakeet's token times: a token that starts with a space starts a word; a word ends where
    the next begins (the last one at the end of the audio)."""
    words = []
    for tok, ts, lp in zip(r.tokens, r.timestamps, r.logprobs or [0.0] * len(r.tokens)):
        if tok.startswith(" ") or not words:
            words.append({"word": tok.strip(), "start": float(ts), "lps": [lp]})
        else:
            words[-1]["word"] += tok
            words[-1]["lps"].append(lp)
    for i, w in enumerate(words):
        w["end"] = round(words[i + 1]["start"] if i + 1 < len(words) else max(duration, w["start"]), 3)
        w["start"] = round(w["start"], 3)
        w["probability"] = round(float(np.exp(np.mean(w.pop("lps")))), 3)
    return [w for w in words if w["word"]]


async def _parakeet_file(path, words):
    model = await get_parakeet()

    def _run():
        from faster_whisper import decode_audio
        audio = decode_audio(path, sampling_rate=16000)
        r = model.recognize(audio, sample_rate=16000)
        text = (r.text or "").strip()
        if not words:
            return text
        duration = round(len(audio) / 16000, 3)
        return {"text": text, "duration": duration, "words": _parakeet_words(r, duration)}

    return await asyncio.get_event_loop().run_in_executor(None, _run)


def _engine(model: Optional[str]) -> str:
    """The speech-to-text engine a request asks for: 'parakeet' or 'whisper' by name; anything else (OpenAI clients
    send 'whisper-1' by habit) gets the default, STT_ENGINE."""
    m = (model or "").strip().lower()
    return "parakeet" if "parakeet" in m else "whisper" if m == "whisper" else STT_ENGINE


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


async def _transcribe_bytes(content: bytes, filename: str, words: bool = False, engine: str = None):
    """The text, or with words=True {text, duration, words: [{word, start, end, probability}]} (seconds into the
    audio, as whisper places them: a phrase's first word can start in the silence before it)."""
    suffix = Path(filename or "rec.webm").suffix or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        engine = engine or STT_ENGINE
        async with _Use(engine):
            if engine == "parakeet":
                return await _parakeet_file(tmp_path, words)
            return await _whisper_file(tmp_path, words)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _whisper_file(tmp_path, words):
    model = await get_whisper()

    def _run():
        segments, info = model.transcribe(tmp_path, beam_size=5, word_timestamps=words)
        if not words:
            return "".join(seg.text for seg in segments).strip()
        segs = list(segments)
        return {"text": "".join(seg.text for seg in segs).strip(), "duration": round(info.duration, 3),
                "words": [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                           "probability": round(w.probability, 3)} for seg in segs for w in (seg.words or [])]}

    return await asyncio.get_event_loop().run_in_executor(None, _run)


def _is_clone(voice: str) -> bool:
    return bool(voice) and (VOICES_DIR / f"{voice}.wav").exists()


async def _clone_wav(text: str, voice: str) -> Optional[bytes]:
    """The cloned voice, or None when its sidecar is down or the GPU is busy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(CLONE_URL + "/v1/audio/speech", json={"input": text, "voice": voice})
        if r.status_code == 200:
            return r.content
        logger.warning("[CLONE] %s: %s", r.status_code, r.text[:120])
    except httpx.HTTPError as e:
        logger.warning("[CLONE] unreachable: %s", e)
    return None


async def _synth_wav(text: str, voice: Optional[str]) -> tuple[bytes, str]:
    """Returns the wav and the engine that made it ("clone" or "kokoro"),
    so a caller can tell a fallback from the real thing."""
    clean = await voice_ready_text(text)
    chosen = voice or DEFAULT_VOICE
    if _is_clone(chosen):
        wav = await _clone_wav(clean, chosen)
        if wav is not None:
            return wav, "clone"
        chosen = KOKORO_VOICE          # the stock voice steps in, same words
    if not chosen or chosen in OPENAI_VOICES:
        chosen = KOKORO_VOICE
    async with _Use("kokoro"):
        pipeline = await get_kokoro()

        def _synth():
            chunks = [audio for _gs, _ps, audio in pipeline(clean, voice=chosen)]
            if not chunks:
                raise RuntimeError("Kokoro produced no audio")
            full = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
            buf = io.BytesIO()
            sf.write(buf, full, KOKORO_SR, format="WAV")
            return buf.getvalue()

        return await asyncio.get_event_loop().run_in_executor(None, _synth), "kokoro"


# ---- the ear ------------------------------------------------------------------
_ear_dl = False


def _new_ear_models():
    """Fresh wake word + VAD models: both keep streaming state, so one pair
    per connection."""
    global _ear_dl
    import openwakeword
    from openwakeword.model import Model
    from openwakeword.vad import VAD
    if not _ear_dl:
        try:
            openwakeword.utils.download_models(model_names=[WAKE_MODEL])
        except Exception as e:  # noqa: BLE001
            logger.warning("[EAR] model download skipped: %s", e)
        _ear_dl = True
    return Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx"), VAD()


class Ear:
    """listening (wake word only) -> capturing (buffer until silence) ->
    transcript; or open (follow-up window, no wake word needed) -> capturing."""

    def __init__(self, oww, vad):
        self.oww, self.vad = oww, vad
        self.state = "listening"
        self.t = 0.0
        self.open_until = 0.0
        self.muted = False
        self.preroll = []
        self.buf = []
        self.capture_t0 = 0.0
        self.last_speech = 0.0
        self.speech_seen = False
        self.source = "wake"

    def open(self, seconds):
        self.open_until = self.t + max(0.0, float(seconds))
        if self.state == "listening":
            self.state = "open"

    def _start(self, source):
        self.state = "capturing"
        self.source = source
        self.buf = list(self.preroll)
        self.capture_t0 = self.t
        self.last_speech = self.t
        self.speech_seen = False

    def _reset(self):
        self.state = "open" if self.t < self.open_until else "listening"
        self.buf = []
        self.speech_seen = False

    def feed(self, frame):
        self.t += EAR_FRAME / EAR_SR
        out = []
        if self.muted:
            return out
        p = float(self.vad.predict(frame))   # __call__ only stores; predict returns the probability
        if self.state == "capturing":
            self.buf.append(frame)
            if p > VAD_THRESHOLD:
                self.last_speech = self.t
                if not self.speech_seen:
                    self.speech_seen = True
                    out.append({"event": "speech_start"})
            dur = self.t - self.capture_t0
            if (self.speech_seen and self.t - self.last_speech > END_SILENCE_S) or dur > MAX_UTTERANCE_S:
                out.append({"event": "speech_end", "seconds": round(dur, 2)})
                out.append(("transcribe", np.concatenate(self.buf), self.source))
                self._reset()
                out.append({"event": "state", "state": self.state})
            elif not self.speech_seen and dur > WAKE_PATIENCE_S:
                out.append({"event": "cancel", "reason": "no speech after the wake word"})
                self._reset()
                out.append({"event": "state", "state": self.state})
        elif self.state == "listening":
            wake = float(self.oww.predict(frame)[WAKE_MODEL])
            if wake > WAKE_THRESHOLD:
                self.oww.reset()
                self._start("wake")
                out.append({"event": "wake", "score": round(wake, 2)})
        else:  # open
            if self.t > self.open_until:
                self.state = "listening"
                out.append({"event": "state", "state": "listening"})
            elif p > VAD_THRESHOLD:
                self._start("followup")
                self.speech_seen = True
                out.append({"event": "speech_start"})
        self.preroll.append(frame)
        del self.preroll[:-PREROLL_FRAMES]
        return out


async def _transcribe_pcm(pcm):
    async with _Use("whisper"):
        model = await get_whisper()
        audio = pcm.astype(np.float32) / 32768.0

        def _run():
            segs, _info = model.transcribe(audio, beam_size=5, language=WHISPER_LANG or None)
            segs = list(segs)
            text = "".join(s.text for s in segs).strip()
            nsp = max((s.no_speech_prob for s in segs), default=1.0)
            lp = min((s.avg_logprob for s in segs), default=-10.0)
            return text, float(nsp), float(lp)

        return await asyncio.get_event_loop().run_in_executor(None, _run)


async def _deliver_transcript(ws, pcm, source):
    start = time.time()
    try:
        text, nsp, lp = await _transcribe_pcm(pcm)
    except Exception as e:  # noqa: BLE001
        logger.error("[EAR] transcription failed: %s", e)
        await ws.send_json({"event": "error", "error": f"transcription failed: {e}"})
        return
    if source == "wake":
        text = WAKE_STRIP.sub("", text, count=1).strip()
    kept = bool(text) and nsp < NO_SPEECH_MAX and lp > LOGPROB_MIN
    logger.info("[EAR] %s %dms nsp=%.2f lp=%.2f %s: %s", source, int((time.time() - start) * 1000), nsp, lp,
                "kept" if kept else "dropped", text[:80])
    await ws.send_json({"event": "transcript", "text": text, "source": source, "kept": kept,
                        "no_speech_prob": round(nsp, 3), "avg_logprob": round(lp, 3),
                        "seconds": round(len(pcm) / EAR_SR, 2)})


app = FastAPI(title="Speakwright voice sidecar")


@app.on_event("startup")
async def _warm():
    """Load the models now rather than on the first word: cold, the first
    transcription took 25 s and the first stock-voice line three minutes on
    the shared GPU. VOICE_WARM=0 keeps the old lazy behavior."""
    asyncio.ensure_future(_idle_watch())
    if os.getenv("VOICE_WARM", "1") != "1":
        return

    async def run():
        for name, step in (("whisper", get_whisper), ("kokoro", get_kokoro)):
            t0 = time.time()
            try:
                await step()
                logger.info("[WARM] %s ready in %ds", name, int(time.time() - t0))
            except Exception as e:  # noqa: BLE001
                logger.warning("[WARM] %s failed: %s", name, e)
        if _is_clone(DEFAULT_VOICE):
            t0 = time.time()
            wav = await _clone_wav("Ready.", DEFAULT_VOICE)
            logger.info("[WARM] clone %s in %ds", "ready" if wav else "unavailable, Kokoro speaks", int(time.time() - t0))

    asyncio.ensure_future(run())


@app.websocket("/v1/audio/stream")
async def audio_stream(ws: WebSocket):
    """Binary frames in: 16 kHz mono PCM16, any size (re-framed to 80 ms).
    Text frames in: {"open": seconds} keeps the ear open without a wake word,
    {"mute": bool} pauses it (while TTS plays). Events out as JSON."""
    await ws.accept()
    try:
        oww, vad = await asyncio.get_event_loop().run_in_executor(None, _new_ear_models)
    except Exception as e:  # noqa: BLE001
        logger.error("[EAR] unavailable: %s", e)
        await ws.send_json({"event": "error", "error": f"ear unavailable: {e}"})
        await ws.close()
        return
    ear = Ear(oww, vad)
    carry = b""
    _state["ears"] += 1                   # an open ear (the VR stage) pins the models: no idle parking meanwhile
    try:
        try:
            async with _idle_lock:
                await asyncio.get_event_loop().run_in_executor(None, _resume, "whisper")
                await asyncio.get_event_loop().run_in_executor(None, _resume, "kokoro")
        except Exception as e:  # noqa: BLE001
            logger.warning("[IDLE] resume on ear open failed: %s", e)
        await ws.send_json({"event": "state", "state": "listening", "wake": WAKE_MODEL})
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                data = carry + msg["bytes"]
                n = len(data) // (EAR_FRAME * 2) * (EAR_FRAME * 2)
                carry = data[n:]
                if not n:
                    continue
                frames = np.frombuffer(data[:n], dtype=np.int16).reshape(-1, EAR_FRAME)
                for fr in frames:
                    for ev in ear.feed(fr):
                        if isinstance(ev, tuple):
                            asyncio.create_task(_deliver_transcript(ws, ev[1], ev[2]))
                        else:
                            await ws.send_json(ev)
            elif msg.get("text"):
                try:
                    ctl = json.loads(msg["text"])
                except ValueError:
                    continue
                if "open" in ctl:
                    ear.open(ctl["open"])
                    await ws.send_json({"event": "state", "state": ear.state})
                if "mute" in ctl:
                    ear.muted = bool(ctl["mute"])
    except WebSocketDisconnect:
        pass
    finally:
        _state["ears"] -= 1
        _state["last_use"] = time.time()


@app.get("/health")
async def health():
    w = ("not loaded" if _whisper is None else _state["whisper_dev"] if _state["whisper_dev"] != "cuda"
         else "gpu" if _whisper.model.model_is_loaded else "parked")
    k = "not loaded" if _kpipe is None else ("parked" if _state["kokoro_parked"] else _state["kokoro_dev"])
    return {"status": "ok", "whisper_loaded": _whisper is not None, "kokoro_loaded": _kpipe is not None,
            "whisper": w, "whisper_compute": _state["whisper_compute"], "kokoro": k, "stt_default": STT_ENGINE,
            "parakeet": "loaded (cpu)" if _parakeet is not None else "not loaded",
            "idle_s": int(time.time() - _state["last_use"]), "idle_unload_s": IDLE_UNLOAD_S,
            "ears_open": _state["ears"], "unloads": _state["unloads"], "resumes": _state["resumes"]}


# ---- OpenAI-compatible endpoints ----------------------------------------------
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form(None),
                                response_format: str = Form(None)):
    start = time.time()
    content = await file.read()
    engine = _engine(model)
    hdr = {"X-STT-Engine": engine}
    try:
        if response_format == "verbose_json":
            out = await _transcribe_bytes(content, file.filename or "rec.webm", words=True, engine=engine)
            logger.info("[STT] %s %dms, %d words: %s", engine, int((time.time() - start) * 1000), len(out["words"]),
                        out["text"][:80])
            return JSONResponse(out, headers=hdr)
        text = await _transcribe_bytes(content, file.filename or "rec.webm", engine=engine)
        logger.info("[STT] %s %dms: %s", engine, int((time.time() - start) * 1000), text[:80])
        return JSONResponse({"text": text}, headers=hdr)
    except HTTPException:
        raise
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
        wav, engine = await _synth_wav(req.input, req.voice)
        logger.info("[TTS] %s: %d bytes in %dms", engine, len(wav), int((time.time() - start) * 1000))
        return Response(content=wav, media_type="audio/wav", headers={"X-Voice-Engine": engine})
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
        wav, engine = await _synth_wav(text, None)
        return Response(content=wav, media_type="audio/wav", headers={"X-Voice-Engine": engine})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")

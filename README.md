# Speakwright

A small local voice sidecar. It does three things:

1. **Ear-readiness rewrite**: an optional LLM pass that rewrites assistant text so it sounds natural read aloud: symbols become words, code blocks get a one-line spoken summary, lists turn into flowing sentences, markdown is dropped. This is the "make the speech right" step (and yes, the name is a pun on it).
2. **Speech-to-text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. **Text-to-speech**: [Kokoro-82M](https://github.com/hexgrad/kokoro), or your own voice cloned by [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) from a short reference clip (see below).

It speaks the **OpenAI audio API**, so any OpenAI-compatible client can point at it. It binds `127.0.0.1` only and needs no API key of its own.

## Endpoints

```
POST /v1/audio/transcriptions   (multipart 'file' [+ 'model'])        -> { text }
POST /v1/audio/speech           ({ input, voice?, response_format? }) -> audio bytes
GET  /health
```

The legacy `/transcribe` and `/tts` routes are kept for backward compatibility.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

For GPU TTS/STT, install a CUDA build of torch into the venv and set `WHISPER_DEVICE=cuda` (Kokoro auto-detects CUDA).

## Run

```bash
.venv/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

On Windows, `start-voice.ps1` starts speakwright and the cloned voice detached with the GPU settings, and leaves alone whatever is already listening:

```powershell
powershell -ExecutionPolicy Bypass -File start-voice.ps1
```

## Config

All optional, see [`.env.example`](.env.example). Set via environment variables:

| Var | Default | Purpose |
|---|---|---|
| `VOICE_PORT` | `8765` | Listen port |
| `VOICE_WARM` | `1` | Load whisper, Kokoro and the clone at startup instead of on the first request |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` (int8) or `cuda`: float16 where the card has it, else int8_float32 (Pascal, GTX 10xx), else float32 |
| `WHISPER_COMPUTE` | _(unset)_ | Force a CTranslate2 compute type, tried first |
| `VOICE_IDLE_UNLOAD_S` | `600` | After this long unused, whisper drops its GPU weights (about 2 s back), Kokoro moves to CPU RAM (about 0.5 s back) and Parakeet is dropped (about 9 s back). An open `/v1/audio/stream` ear pins them. `0` turns it off |
| `STT_ENGINE` | `whisper` | Default speech-to-text engine for files: `whisper` or `parakeet`. A request picks one with `model=parakeet` or `model=whisper`; `whisper-1` gets the default. The ear always uses whisper |
| `PARAKEET_MODEL` / `PARAKEET_THREADS` | `nemo-parakeet-tdt-0.6b-v2` / `2` | Parakeet via onnx-asr, int8 on the CPU (`uv pip install --no-deps onnx-asr`; the 631 MB model downloads on first use) |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice, and the fallback when a cloned voice cannot speak |
| `DEFAULT_VOICE` | `KOKORO_VOICE` | Voice used when a request names none; set it to a clip name to speak in that voice |
| `VOICE_CLONE_URL` | `http://127.0.0.1:8766` | Where clone.py listens |
| `KOKORO_LANG` | `a` | Kokoro language code |
| `KOKORO_DEVICE` | auto | Force `cuda`/`cpu` |
| `VOICE_AGENT_MODEL` | _(unset)_ | Enables the ear-readiness rewrite when set with a key |
| `VOICE_AGENT_BASE_URL` | OpenRouter | Any OpenAI-compatible chat endpoint |
| `VOICE_AGENT_API_KEY` | _(unset)_ | Key for the rewrite model |

Without `VOICE_AGENT_MODEL` + `VOICE_AGENT_API_KEY`, TTS still works using a regex baseline cleanup instead of the LLM rewrite.

## Your own voice

Drop a reference clip at `voices/<name>.wav` (10 to 20 seconds of one person talking, mono, no music) and any request with `voice: "<name>"` is synthesized in that voice. The cloning model needs an older torch than the rest of the sidecar, so it runs as a second process from its own venv:

```bash
uv venv --python 3.11 .venv-clone
uv pip install --python .venv-clone --index-url https://download.pytorch.org/whl/cu126 "torch==2.6.0" "torchaudio==2.6.0"
uv pip install --python .venv-clone -r requirements-clone.txt
.venv-clone/Scripts/python clone.py            # 127.0.0.1:8766
```

The first request downloads Chatterbox-Turbo (about 2 GB) into `.hf/` next to the script and loads it. `DEFAULT_VOICE=<name>` makes the clone the voice of every request that does not name one. When clone.py is down, or the GPU is busy with a render and there is no room for the model, the same words come out in the Kokoro voice instead, so speech never fails because the clone could not run. Reference clips and the weights are git-ignored: a voice print stays on the machine it was recorded on.

| Var | Default | Purpose |
|---|---|---|
| `CLONE_PORT` | `8766` | Listen port |
| `CLONE_DEVICE` | `cuda` | `cuda` or `cpu` (slow) |
| `CLONE_DEFAULT` | `nate` | Clip used when a request names no voice |
| `CLONE_MIN_FREE_GB` | `1.2` | Below this much free VRAM the clone refuses and Kokoro speaks |
| `HF_HOME` | `.hf/` | Where the weights download |

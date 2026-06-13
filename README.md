# Speakwright

A small local voice sidecar. It does three things:

1. **Ear-readiness rewrite** — an optional LLM pass that rewrites assistant text so it sounds natural read aloud: symbols become words, code blocks get a one-line spoken summary, lists turn into flowing sentences, markdown is dropped. This is the "make the speech right" step (and yes, the name is a pun on it).
2. **Speech-to-text** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. **Text-to-speech** — [Kokoro-82M](https://github.com/hexgrad/kokoro).

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

## Config

All optional — see [`.env.example`](.env.example). Set via environment variables:

| Var | Default | Purpose |
|---|---|---|
| `VOICE_PORT` | `8765` | Listen port |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` (int8) or `cuda` (float16) |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice |
| `KOKORO_LANG` | `a` | Kokoro language code |
| `KOKORO_DEVICE` | auto | Force `cuda`/`cpu` |
| `VOICE_AGENT_MODEL` | _(unset)_ | Enables the ear-readiness rewrite when set with a key |
| `VOICE_AGENT_BASE_URL` | OpenRouter | Any OpenAI-compatible chat endpoint |
| `VOICE_AGENT_API_KEY` | _(unset)_ | Key for the rewrite model |

Without `VOICE_AGENT_MODEL` + `VOICE_AGENT_API_KEY`, TTS still works using a regex baseline cleanup instead of the LLM rewrite.

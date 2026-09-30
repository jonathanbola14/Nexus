# AGENTS.md

Voice assistant ("nexus") — wake-word → ASR → LLM agent (LangChain + NVIDIA NIM) → streaming TTS (Kokoro). Single-process Python 3.11 app; implementation modules are under `src/`, tests under `tests/`, and `main.py`/`preview.py` are the entrypoints.

## Run

`src/LLM.py` reads `NVIDIA_API_KEY` from the environment first, then from the project-root `.env` using `python-dotenv`. Either source is sufficient.

```bash
export NVIDIA_API_KEY="..."
uv run python main.py
```

Requires Python 3.11 (pinned `>=3.11,<3.12` in `pyproject.toml`); the `torch` CPU wheel is pinned via `[tool.uv.sources]` to a specific URL — don't let uv resolve a CUDA/different torch, TTS depends on it.

## Architecture (entrypoint = `main.py`)

`main.py` owns the capture loop and wires the implementation modules under `src/` together; do not treat them as interchangeable services:

- `src/WakeWord.py` — openWakeWord (`models/nexus.onnx`) on a 16 kHz mic stream; on activation records speech via `src/utils/recorder.py` (webrtcvad + silence-frames).
- `src/STT.py` — `onnx_asr` with `istupakov/parakeet-tdt-0.6b-v3-onnx` (int8). Model is a module-level singleton (`_stt_model`); call `load_stt_model()`, never re-instantiate.
- `src/LLM.py` — DeepAgents + `ChatNVIDIA` (`z-ai/glm-5.3-flash`). Agent tools are defined under `src/tools/`; system prompt is pt-BR, so keep responses in Portuguese.

Streaming TTS pipeline: LLM tokens stream into `text_buffer` → `split_ready_sentences()` splits on real sentence boundaries (handles pt-BR abbreviations and decimal `.` mid-stream) → `tts_queue` → `Kokoro` (`KPipeline(lang_code="p", voice="pm_alex")`, 24 kHz) → `playback_queue` → `src/utils/play_file.py` playback.

## Concurrency gotchas (these bit before — comments in `main.py` explain)

- `print_lock` is required around **all** stdout/stderr writes. `tts_worker` wraps the Kokoro call in `redirect_stdout`/`redirect_stderr` globally (per-process, not per-thread); without the lock, the LLM streaming prints on the main thread disappear while TTS runs.
- A fresh `tts_thread`/`playback_thread` pair is created **every turn**. Old threads have already exited (they consume `STOP_SIGNAL` and can't restart). Don't try to reuse them.
- `STOP_SIGNAL` **must** always be pushed onto `tts_queue`, even on exception — the `finally` block in `main.py:283` does this. Skipping it leaks a thread blocking on the queue forever, and the next turn's duplicate consumer fights over items.
- `text_buffer`/`sentence_buffer`/header flags are reset per turn inside the loop; don't hoist them out (turn N's text would bleed into turn N+1).

## Setup quirks

- `audios/noise.wav` is recorded on first run (3.5 s of silence, amplified 2×) — the app prompts "fique em silencio...". It's committed-ish (regenerated if missing) and is a runtime dependency of `STT.py` (`nr.reduce_noise` uses it as the noise profile). Don't delete it.
- `audios/activation.wav` and `audios/beep.wav` are required assets (wake-activation beep + post-record confirmation).
- PyAudio opens the default input/output device at import time in multiple places (`main.py`, `src/WakeWord.py`, `src/utils/play_file.py`). Real audio hardware is assumed; CI/headless won't work.
- `models/nexus.onnx` (+ `embedding_model.onnx`, `melspectrogram.onnx`) must exist or `configWakeWord` calls `sys.exit()`.

## Style

- All user-facing strings and comments are **Portuguese**. Match this when editing existing files.
- `setuptools<81` pin in `requirements.txt`/`pyproject.toml` is deliberate (compat with pyaudio/older build) — don't bump casually.

## ⚠️ Security note

`.env` is ignored but already tracked by Git, so ignore rules do not protect its contents or remove it from history. Do not expose its contents; rotate any real key and remove the file from version control before sharing the repository.



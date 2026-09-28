# Nexus

A personal voice assistant designed for terminal-based use. Nexus detects a wake word, transcribes spoken Portuguese, queries a NVIDIA NIM-powered agent, and responds using Brazilian Portuguese speech synthesis.

Audio capture, speech recognition, and voice synthesis all run locally. The agent response uses the NVIDIA API and requires a valid access key.

## Requirements

- Linux x86_64 and Python 3.12.
- [uv](https://docs.astral.sh/uv/) to install and run the project environment.
- A working microphone and audio output device.
- System audio libraries required to compile or run PyAudio, including PortAudio.
- A NVIDIA API key with access to the model configured for the agent.

PyTorch is pinned to a CPU wheel for Python 3.12 on Linux x86_64 in `pyproject.toml`.

## Installation

From the project root, install the dependencies:

```bash
uv sync
```

Configure the API key in a `.env` file in the project root:

```dotenv
NVIDIA_API_KEY=sua-chave-nvidia
```

Keep this file private and do not publish or share the key.

## Usage

To start with wake-word detection enabled:

```bash
uv run python main.py
```

To start speech capture without waiting for the wake word:

```bash
uv run python main.py --sem-ativacao
```

Stop the process with `Ctrl+C`. On the first run, if `audios/noise.wav` does not exist, the application records a few seconds of ambient noise for noise reduction. Remain silent during this step.

## Processing flow

1. `WakeWord.py` detects the activation phrase using `models/nexus.onnx`. With `--sem-ativacao`, this step is skipped.
2. `utils/recorder.py` captures the spoken input and ends recording after silence is detected.
3. `STT.py` reduces noise and transcribes the audio using the Parakeet ONNX model configured for Brazilian Portuguese.
4. `LLM.py` sends the user message to the NVIDIA agent, which includes tools for checking the current date and time.
5. `main.py` receives the streamed response and splits it into sentences so they can be synthesized while the remainder is still being generated.
6. The audio is synthesized by Kokoro and played through the system’s default output device.

## Required files

- `models/nexus.onnx`, `models/embedding_model.onnx`, and `models/melspectrogram.onnx` for wake-word detection.
- `audios/activation.wav` and `audios/beep.wav` for the audio prompts and alert tones.
- `audios/noise.wav` as the noise profile; it is created on the first run if it is missing.

The project depends on real audio hardware and is not designed to run without a microphone or in headless environments. There is no automated test suite configured at this time.
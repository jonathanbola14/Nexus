"""Wake-word detection and transition into speech recording."""

import collections
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyaudio
from openwakeword import Model as ModelWakeWord
from pyaudio import PyAudio
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from src.utils.play_file import Play
from src.utils.audio import resample_pcm16
from src.utils.recorder import collect_speech_frames

WAKEWORD_RATE = 16000


def configWakeWord(
    mic: PyAudio,
    RATE: int,
    model_path="models/nexus.onnx",
    threshold=0.65,
    input_rate: int | None = None,
):
    """Open the detector stream and initialize the wake-word model state."""
    input_rate = input_rate or RATE
    wake_stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=input_rate,
        input=True,
        frames_per_buffer=int(input_rate * 0.08)
    )
    # The detector consumes audio continuously until the activation is stable.
    last_save = time.time()
    activation_times = collections.defaultdict(list)
    # Keep a short context window after activation before recording speech.
    save_delay = 0.3  # seconds

    # Prevent a single utterance from triggering multiple recordings.
    cooldown = 4  # seconds

    if model_path and os.path.exists(model_path):
        owwModel = ModelWakeWord(
            wakeword_model_paths=[model_path],
            enable_speex_noise_suppression=False,
            vad_threshold=threshold,
        )
    else:
        print(f'Could not find model "{model_path}"')
        sys.exit()

    return wake_stream, last_save, activation_times, save_delay, cooldown, owwModel


def _audio_path(filename: str) -> str:
    """Resolve an audio asset path relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "audios" / filename)


def WakeWord(
    wake_stream: pyaudio.Stream,
    RATE: int,
    CHUNK: int,
    stream: pyaudio.Stream,
    owwModel: ModelWakeWord,
    activation_times,
    last_save,
    cooldown,
    save_delay,
    player: Play,
    live: Live,
    threshold=0.65,
    input_rate: int | None = None,
):
    """Read a detector frame and record speech when the wake word activates."""
    input_rate = input_rate or RATE
    # Resample the microphone frame to the detector's required sample rate.
    mic_audio = np.frombuffer(
        buffer=resample_pcm16(
            wake_stream.read(
                num_frames=int(input_rate * 0.08),
                exception_on_overflow=False,
            ),
            input_rate,
            WAKEWORD_RATE,
        ),
        dtype=np.int16,
    )

    # Feed the frame to openWakeWord.
    prediction = owwModel.predict(mic_audio)

    # Track scores above the configured activation threshold.
    for mdl in prediction:
        if prediction[mdl] >= threshold:
            activation_times[mdl].append(time.time())

        if activation_times.get(mdl) and (time.time() - last_save) >= cooldown \
                and (time.time() - activation_times.get(mdl)[0]) >= save_delay:
            last_save = time.time()
            activation_times[mdl] = []

            message = f"Palavra de ativação detectada: {mdl}"
            if getattr(live, "_is_textual_bridge", False) is True:
                live.set_status(message, "ATIVAÇÃO")
                
            else:
                live.update(
                    Panel(
                        Text(message, style="bold green"),
                        title="[bold green] NEXUS [/bold green] [dim]/[/dim] [bold]ATIVAÇÃO[/bold]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

            # Reuse the player created by main.py. Creating a player for each
            # activation would also create output streams that are never closed.
            player.file(file=_audio_path("activation.wav"))
            time.sleep(0.15)

            audio = collect_speech_frames(
                stream, RATE, CHUNK, live, input_rate=input_rate
            )

            # Discard frames captured during speech to avoid immediate retriggers.
            wake_stream.read(
                num_frames=wake_stream.get_read_available(),
                exception_on_overflow=False,
            )
            # Clear the model's rolling prediction state before the next utterance.
            owwModel.reset()

            return audio, last_save, activation_times

    # Return None when no activation occurred so the caller can skip STT. Empty
    # bytes would otherwise reach the recognizer as a zero-length waveform.
    return None, last_save, activation_times

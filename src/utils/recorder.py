import pyaudio
import webrtcvad
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from src.utils.audio import resample_pcm16

# from .increase_gain import gain

vad = webrtcvad.Vad(2)


def _show_capture_status(live: Live, message: str, style: str):
    if getattr(live, "_is_textual_bridge", False) is True:
        state = "CAPTURANDO" if style == "bold green" else "OUVINDO"
        if "Transcrevendo" in message:
            state = "PENSANDO"
        live.set_status(message, state)
        return

    live.update(
        Panel(
            Text(message, style=style),
            title="[bold green] NEXUS [/bold green] [dim]/[/dim] [bold]MICROFONE[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )


def collect_speech_frames(
    stream: pyaudio.Stream,
    RATE: int,
    CHUNK: int,
    live: Live,
    input_rate: int | None = None,
):
    input_rate = input_rate or RATE
    frames = []
    silence = 0
    has_speech = False
    capture_state = "waiting"
    _show_capture_status(live, "Aguardando você falar...", "dim")

    while True:
        frame = stream.read(num_frames=CHUNK, exception_on_overflow=False)
        frame = resample_pcm16(frame, input_rate, RATE)
        # frame = gain(frame, ganho=1.5)

        if vad.is_speech(frame, RATE):
            frames.append(frame)
            silence = 0
            has_speech = True
            if capture_state != "capturing":
                _show_capture_status(live, "Capturando sua fala...", "bold green")
                capture_state = "capturing"

        else:
            if not has_speech:
                continue

            silence += 1
            frames.append(frame)
            if has_speech and capture_state != "silence":
                _show_capture_status(live, "Pausa detectada; aguardando retomada...", "bold yellow")
                capture_state = "silence"

            if silence >= 30:
                break

    _show_capture_status(live, "Transcrevendo fala...", "bold cyan")

    return b"".join(frames)
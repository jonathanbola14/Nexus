"""Audio playback helpers for WAV files and in-memory synthesized speech."""

import wave

import numpy as np
import pyaudio


class Play:
    """Manage the PyAudio output stream used by the assistant."""

    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()

        self.format = self.p.get_format_from_width(2)

        try:
            device_info = self.p.get_default_output_device_info()
        except Exception:
            device_info = {}

        default_sample_rate = int(device_info.get("defaultSampleRate", 48000))
        default_channels = int(device_info.get("maxOutputChannels", 2) or 2)
        if default_channels <= 0:
            default_channels = 2

        self.channels = default_channels
        self.sample_rate = default_sample_rate

        self.file_path = ""
        self.wf: wave.Wave_read | None = None

        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
        )

    def file(self, file: str):
        """Play a WAV file, reopening the stream when its format differs."""
        self.file_path = file
        self.wf = wave.open(self.file_path, "rb")

        fmt = self.p.get_format_from_width(self.wf.getsampwidth())
        ch = self.wf.getnchannels()
        rate = self.wf.getframerate()

        # PyAudio streams retain their original format, so changed WAV settings
        # require reopening the stream rather than updating attributes alone.
        if fmt != self.format or ch != self.channels or rate != self.sample_rate:
            self.stream.stop_stream()
            self.stream.close()
            self.format = fmt
            self.channels = ch
            self.sample_rate = rate
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
            )

        dados = self.wf.readframes(1024)
        while dados:
            self.stream.write(dados)
            dados = self.wf.readframes(1024)

    def tensor(self, audio, samplerate: int = 24000, channels: int = 1):
        """Play a synthesized waveform, accepting NumPy or tensor input."""
        # Kokoro returns a torch.Tensor; normalize it to a NumPy array first.
        if not isinstance(audio, np.ndarray):
            audio = (
                audio.detach().cpu().numpy()
                if hasattr(audio, "detach")
                else np.asarray(audio)
            )

        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        self.format = self.p.get_format_from_width(2)

        # Reopen the stream when the requested format differs; changing these
        # attributes alone does not update PyAudio's actual stream settings.
        if samplerate != self.sample_rate or channels != self.channels:
            self.stream.stop_stream()
            self.stream.close()
            self.sample_rate = samplerate
            self.channels = channels
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
            )

        audio_bytes = audio.tobytes()
        chunk_size = 1024 * 2

        for offset in range(0, len(audio_bytes), chunk_size):
            self.stream.write(audio_bytes[offset : offset + chunk_size])

    def stop(self):
        """Close the output stream, PyAudio instance, and open WAV file."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        if self.wf is not None:
            self.wf.close()
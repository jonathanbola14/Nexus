import wave

import numpy as np
import pyaudio


class Play:
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()

        self.format = self.p.get_format_from_width(2)
        self.channels = 2
        self.sample_rate = 16000

        self.file_path = ""
        self.wf: wave.Wave_read | None = None  # <-- tipo correto, não o módulo

        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
        )

    def file(self, file: str):
        self.file_path = file
        self.wf = wave.open(self.file_path, "rb")

        fmt = self.p.get_format_from_width(self.wf.getsampwidth())
        ch = self.wf.getnchannels()
        rate = self.wf.getframerate()

        # O stream é aberto no __init__ com 2 canais/16 kHz. Sem reabrir, o
        # PyAudio toca com a configuração antiga mesmo que format/sample_rate/
        # channels do WAV sejam outros — som distorcido pelo sample_rate
        # errado. Espelha a lógica já existente em tensor().
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
        # O Kokoro retorna um torch.Tensor, não um numpy.ndarray — converte antes de tudo
        if not isinstance(audio, np.ndarray):
            audio = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)

        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        self.format = self.p.get_format_from_width(2)

        # O stream é aberto uma vez no __init__ com outra taxa/canais; se o
        # áudio pedido usa valores diferentes, precisa reabrir o stream —
        # só trocar os atributos não muda a configuração real do PyAudio.
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

        dados = audio.tobytes()
        chunk_size = 1024 * 2

        for i in range(0, len(dados), chunk_size):
            self.stream.write(dados[i:i + chunk_size])

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        if self.wf is not None:
            self.wf.close()
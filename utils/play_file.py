import wave

import numpy as np
import pyaudio


class Play:
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()

        self.format = 0
        self.channels = 0
        self.sample_rate = 0
        
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
        )

    def file(self, file: str):

        self.wf = wave.open(file, "rb")

        self.format = self.p.get_format_from_width(self.wf.getsampwidth())
        self.channels=self.wf.getnchannels()
        self.rate=self.wf.getframerate()

        dados = self.wf.readframes(1024)

        while dados:
            self.stream.write(dados)
            dados = self.wf.readframes(1024)

    

    def tensor(self, audio: np.ndarray, samplerate: int = 24000, channels: int = 1):
        # Garante float32 no range [-1, 1] antes de converter para int16 (PCM)
        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        self.format = self.p.get_format_from_width(2),  # 2 bytes = 16 bits (int16)
        self.sample_rate = samplerate
        self.channels = channels

        dados = audio.tobytes()
        chunk_size = 1024 * 2  # 1024 amostras * 2 bytes por amostra (int16)

        for i in range(0, len(dados), chunk_size):
            self.stream.write(dados[i:i + chunk_size])

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()
        self.wf.close()
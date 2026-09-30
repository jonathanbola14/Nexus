import numpy as np


def gain(frame: bytes, ganho: float) -> bytes:
    audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)

    audio *= ganho

    # Evita distorção por overflow
    audio = np.clip(audio, -32768, 32767)

    return audio.astype(np.int16).tobytes()

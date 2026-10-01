import numpy as np


def gain(frame: bytes, ganho: float) -> bytes:
    """Scale signed 16-bit PCM samples and clip them to the valid range."""
    audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)

    audio *= ganho

    # Clipping prevents overflow distortion when the gain exceeds full scale.
    audio = np.clip(audio, -32768, 32767)

    return audio.astype(np.int16).tobytes()

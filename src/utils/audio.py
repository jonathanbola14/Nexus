"""Audio format conversion helpers shared by capture and detection."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample mono signed 16-bit PCM bytes without changing their format."""
    if source_rate == target_rate or not data:
        return data

    samples = np.frombuffer(data, dtype=np.int16)
    divisor = gcd(source_rate, target_rate)
    converted = resample_poly(
        samples,
        target_rate // divisor,
        source_rate // divisor,
    )
    converted = np.clip(np.rint(converted), -32768, 32767).astype(np.int16)
    return converted.tobytes()
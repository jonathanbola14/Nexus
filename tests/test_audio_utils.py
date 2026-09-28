import numpy as np

from utils.increase_gain import gain


def test_gain_scales_audio_samples():
    frame = np.array([-1000, 0, 1000], dtype=np.int16).tobytes()

    result = np.frombuffer(gain(frame, 2), dtype=np.int16)

    np.testing.assert_array_equal(result, [-2000, 0, 2000])


def test_gain_clips_samples_to_int16_range():
    frame = np.array([-20000, 20000], dtype=np.int16).tobytes()

    result = np.frombuffer(gain(frame, 2), dtype=np.int16)

    np.testing.assert_array_equal(result, [-32768, 32767])


def test_gain_accepts_empty_audio_frame():
    assert gain(b"", 2) == b""
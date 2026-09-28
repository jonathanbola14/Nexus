import numpy as np

from utils.increase_gain import gain
from utils.audio import resample_pcm16
from utils.play_file import Play


def test_play_uses_default_output_sample_rate(monkeypatch):
    opened = {}

    class DummyStream:
        def stop_stream(self):
            pass

        def close(self):
            pass

    class DummyPyAudio:
        def get_format_from_width(self, width):
            return "format"

        def get_default_output_device_info(self):
            return {"defaultSampleRate": 44100}

        def open(self, **kwargs):
            opened.update(kwargs)
            return DummyStream()

    monkeypatch.setattr("utils.play_file.pyaudio.PyAudio", DummyPyAudio)

    player = Play()

    assert player.sample_rate == 44100
    assert opened["rate"] == 44100


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


def test_resample_pcm16_converts_44100_hz_frame_to_16000_hz():
    source = np.zeros(1323, dtype=np.int16).tobytes()

    result = np.frombuffer(resample_pcm16(source, 44100, 16000), dtype=np.int16)

    assert len(result) == 480
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np


def load_wakeword_module(monkeypatch, collect_speech_frames):
    pyaudio_module = types.ModuleType("pyaudio")
    pyaudio_module.paInt16 = 8
    pyaudio_module.PyAudio = object
    pyaudio_module.Stream = object
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio_module)

    openwakeword_module = types.ModuleType("openwakeword")
    openwakeword_module.Model = object
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword_module)

    src_module = types.ModuleType("src")
    src_module.__path__ = []
    monkeypatch.setitem(sys.modules, "src", src_module)
    utils_module = types.ModuleType("src.utils")
    utils_module.__path__ = []
    monkeypatch.setitem(sys.modules, "src.utils", utils_module)
    play_file_module = types.ModuleType("src.utils.play_file")
    play_file_module.Play = object
    recorder_module = types.ModuleType("src.utils.recorder")
    recorder_module.collect_speech_frames = collect_speech_frames
    monkeypatch.setitem(sys.modules, "src.utils.play_file", play_file_module)
    monkeypatch.setitem(sys.modules, "src.utils.recorder", recorder_module)

    module_path = Path(__file__).parents[1] / "src" / "WakeWord.py"
    module_spec = importlib.util.spec_from_file_location("wakeword_under_test", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_wakeword_returns_none_when_prediction_is_below_threshold(monkeypatch):
    collect_speech_frames = Mock()
    module = load_wakeword_module(monkeypatch, collect_speech_frames)
    wake_stream = Mock()
    wake_stream.read.return_value = np.array([1, 2], dtype=np.int16).tobytes()
    wake_model = Mock()
    wake_model.predict.return_value = {"nexus": 0.4}
    activation_times = {}

    result = module.WakeWord(
        wake_stream=wake_stream,
        RATE=16000,
        CHUNK=480,
        stream=Mock(),
        owwModel=wake_model,
        activation_times=activation_times,
        last_save=0,
        cooldown=4,
        save_delay=0.3,
        player=Mock(),
        live=Mock(),
        threshold=0.65,
    )

    assert result == (None, 0, activation_times)
    assert activation_times == {}
    collect_speech_frames.assert_not_called()
    wake_model.reset.assert_not_called()


def test_wakeword_predicts_at_16khz_when_application_rate_differs(monkeypatch):
    module = load_wakeword_module(monkeypatch, Mock())
    wake_stream = Mock()
    wake_stream.read.return_value = np.zeros(1280, dtype=np.int16).tobytes()
    wake_model = Mock()
    wake_model.predict.return_value = {"nexus": 0.4}

    module.WakeWord(
        wake_stream=wake_stream,
        RATE=20000,
        CHUNK=600,
        stream=Mock(),
        owwModel=wake_model,
        activation_times={},
        last_save=0,
        cooldown=4,
        save_delay=0.3,
        player=Mock(),
        live=Mock(),
        threshold=0.65,
        input_rate=16000,
    )

    assert wake_model.predict.call_args.args[0].size == 1280


def test_wakeword_records_after_threshold_and_delay(monkeypatch):
    recorded_audio = b"recorded audio"
    collect_speech_frames = Mock(return_value=recorded_audio)
    module = load_wakeword_module(monkeypatch, collect_speech_frames)
    time_values = iter([10.0, 15.0, 15.4, 15.5])
    monkeypatch.setattr(module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(module.time, "sleep", Mock())

    wake_stream = Mock()
    wake_stream.read.side_effect = [
        np.array([1, 2], dtype=np.int16).tobytes(),
        b"queued frames",
    ]
    wake_stream.get_read_available.return_value = 64
    wake_model = Mock()
    wake_model.predict.return_value = {"nexus": 0.9}
    player = Mock()
    live = Mock()
    audio_stream = Mock()
    activation_times = {"nexus": []}

    result = module.WakeWord(
        wake_stream=wake_stream,
        RATE=16000,
        CHUNK=480,
        stream=audio_stream,
        owwModel=wake_model,
        activation_times=activation_times,
        last_save=10.0,
        cooldown=4,
        save_delay=0.3,
        player=player,
        live=live,
        threshold=0.65,
    )

    assert result == (recorded_audio, 15.5, {"nexus": []})
    player.file.assert_called_once()
    collect_speech_frames.assert_called_once_with(
        audio_stream, 16000, 480, live, input_rate=16000
    )
    wake_stream.read.assert_called_with(num_frames=64, exception_on_overflow=False)
    wake_model.reset.assert_called_once()
    live.update.assert_called_once()
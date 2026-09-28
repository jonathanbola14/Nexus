from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import STT


def test_load_stt_model_caches_loaded_model(monkeypatch):
    loaded_model = object()
    load_model = Mock(return_value=loaded_model)
    monkeypatch.setattr(STT, "_stt_model", None)
    monkeypatch.setattr(STT.onnx_asr, "load_model", load_model)

    assert STT.load_stt_model() is loaded_model
    assert STT.load_stt_model() is loaded_model
    load_model.assert_called_once_with(
        model="istupakov/parakeet-tdt-0.6b-v3-onnx",
        quantization="int8",
    )


def test_load_stt_model_falls_back_without_quantization(monkeypatch):
    loaded_model = object()
    load_model = Mock(side_effect=[RuntimeError("int8 indisponível"), loaded_model])
    monkeypatch.setattr(STT, "_stt_model", None)
    monkeypatch.setattr(STT.onnx_asr, "load_model", load_model)

    assert STT.load_stt_model() is loaded_model
    assert load_model.call_count == 2
    assert load_model.call_args_list[1].kwargs == {
        "model": "istupakov/parakeet-tdt-0.6b-v3-onnx"
    }


def test_speech_to_text_reduces_noise_and_passes_sample_rate(monkeypatch):
    noise_samples = np.array([5, -7], dtype=np.int16)

    class NoiseFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getnframes(self):
            return 12

        def readframes(self, nframes):
            assert nframes == 2
            return noise_samples.tobytes()

    monkeypatch.setattr(STT.wave, "open", lambda **kwargs: NoiseFile())
    reduce_noise = Mock(side_effect=lambda **kwargs: kwargs["y"])
    monkeypatch.setattr(STT.nr, "reduce_noise", reduce_noise)
    model = SimpleNamespace(recognize=Mock(return_value="transcrição"))
    audio_bytes = np.array([100, -200], dtype=np.int16).tobytes()

    result = STT.Speech_to_Text(audio_bytes, RATE=16000, model=model)

    assert result == "transcrição"
    reduce_noise.assert_called_once()
    np.testing.assert_array_equal(
        reduce_noise.call_args.kwargs["y_noise"],
        noise_samples.astype(np.float32),
    )
    model.recognize.assert_called_once()
    recognized_audio = model.recognize.call_args.args[0]
    np.testing.assert_array_equal(recognized_audio, [100.0, -200.0])
    assert model.recognize.call_args.kwargs == {
        "sample_rate": 16000,
        "target_language": "pt-br",
    }
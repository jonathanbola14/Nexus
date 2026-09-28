import wave
from pathlib import Path

import noisereduce as nr
import numpy as np
import onnx_asr

_stt_model = None


def load_stt_model():
    """Carrega o modelo ASR uma única vez (evita recarregar a cada chamada)."""
    global _stt_model
    if _stt_model is None:
        try:
            _stt_model = onnx_asr.load_model(
                model="istupakov/parakeet-tdt-0.6b-v3-onnx",
                quantization='int8',
            )
        except Exception:
            _stt_model = onnx_asr.load_model(
                model="istupakov/parakeet-tdt-0.6b-v3-onnx",
            )

    return _stt_model


def _audio_path(filename: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "audios" / filename)


def Speech_to_Text(data: bytes, RATE: int, model=None):
    if model is None:
        model = load_stt_model()

    # bytes -> numpy
    audio = np.frombuffer(buffer=data, dtype=np.int16).astype(np.float32)

    with wave.open(_audio_path("noise.wav"), mode='rb') as f:
        nframes = f.getnframes()
        ruido = f.readframes(nframes=max(0, nframes - 10))

    ruido = np.frombuffer(buffer=ruido, dtype=np.int16).astype(np.float32)

    # reduz ruído
    if len(ruido) > 0:
        audio = nr.reduce_noise(
            y=audio,
            sr=RATE,
            stationary=False,
            y_noise=ruido,
            n_jobs=-1,
            use_tqdm=True,
            n_fft=1024
        )

    # Recognize exige sample_rate ao receber waveform numpy (não path).
    # Sem ele, o modelo pode usar um default incorreto e transcrever errado.
    result = model.recognize(audio, sample_rate=RATE, target_language="pt-br")
    return result

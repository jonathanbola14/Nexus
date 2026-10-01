"""Speech-to-text model loading and audio transcription helpers."""

import wave
from pathlib import Path

import noisereduce as nr
import numpy as np
import onnx_asr

_stt_model = None


def load_stt_model():
    """Load the ASR model once and reuse the cached instance."""
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
    """Resolve an audio asset path relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "audios" / filename)


def Speech_to_Text(data: bytes, RATE: int, model=None):
    """Reduce background noise and transcribe signed 16-bit PCM audio."""
    if model is None:
        model = load_stt_model()

    # Convert signed 16-bit PCM bytes to the float waveform expected by ASR.
    audio = np.frombuffer(buffer=data, dtype=np.int16).astype(np.float32)

    with wave.open(_audio_path("noise.wav"), mode='rb') as f:
        nframes = f.getnframes()
        noise_bytes = f.readframes(nframes=max(0, nframes - 10))

    noise_audio = np.frombuffer(buffer=noise_bytes, dtype=np.int16).astype(np.float32)

    if len(noise_audio) > 0:
        audio = nr.reduce_noise(
            y=audio,
            sr=RATE,
            stationary=False,
            y_noise=noise_audio,
            n_jobs=-1,
            use_tqdm=True,
            n_fft=1024
        )

    # Waveform input requires an explicit sample rate; otherwise the model may
    # assume a different rate and produce an incorrect transcription.
    result = model.recognize(audio, sample_rate=RATE, target_language="pt-br")
    return result

import collections
import os
import sys
import time

import numpy as np
import pyaudio
from openwakeword import Model as ModelWakeWord
from pyaudio import PyAudio

from utils.play_file import Play
from utils.recorder import collect_speech_frames


def configWakeWord(mic: PyAudio, RATE: int, model_path="models/nexus.onnx", threshold=0.65):
    wake_stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=1280
    )
    # Predict continuously on audio stream
    last_save = time.time()
    activation_times = collections.defaultdict(list)
    # Set waiting period after activation before saving clip (to get some audio context after the activation)
    save_delay = 0.3  # seconds

    # Set cooldown period before another clip can be saved
    cooldown = 4  # seconds

    if model_path and os.path.exists(model_path):
        owwModel = ModelWakeWord(
            wakeword_model_paths=[model_path],
            enable_speex_noise_suppression=False,
            vad_threshold=threshold,
        )
    else:
        print(f'Could not find model "{model_path}"')
        sys.exit()

    return wake_stream, last_save, activation_times, save_delay, cooldown, owwModel


def WakeWord(
    wake_stream: pyaudio.Stream,
    RATE: int,
    CHUNK: int,
    stream: pyaudio.Stream,
    owwModel: ModelWakeWord,
    activation_times,
    last_save,
    cooldown,
    save_delay,
    player: Play,
    threshold=0.65,
):
    # Get audio
    mic_audio = np.frombuffer(
        buffer=wake_stream.read(num_frames=1280, exception_on_overflow=False),
        dtype=np.int16,
    )

    # Feed to openWakeWord model
    prediction = owwModel.predict(mic_audio)

    # Check for model activations (score above threshold)
    for mdl in prediction:
        if prediction[mdl] >= threshold:
            activation_times[mdl].append(time.time())

        if activation_times.get(mdl) and (time.time() - last_save) >= cooldown \
                and (time.time() - activation_times.get(mdl)[0]) >= save_delay:
            last_save = time.time()
            activation_times[mdl] = []

            print(f'\n\nDetected activation from "{mdl}" model at time!')

            # Reusa o player global (instanciado uma vez em main.py) em vez de
            # criar um novo Play() — cada Play() abre um PyAudio() + output
            # stream e nunca os fecharíamos, vazando recursos a cada wake word.
            player.file(file=os.path.join(os.path.dirname(__file__), 'audios', 'activation.wav'))
            time.sleep(0.15)

            print()

            audio = collect_speech_frames(stream, RATE, CHUNK)

            # Evita reativação: drena frames acumulados no wake_stream durante a fala
            wake_stream.read(num_frames=wake_stream.get_read_available(), exception_on_overflow=False)
            # Limpa o buffer de previsões do modelo openWakeWord
            owwModel.reset()

            return audio, last_save, activation_times

    # Sem áudio: retorna explicitamente None para o main.py poder filtrar com
    # `if audio is None`. Antes retornava bytes(0), que passava no filtro e
    # chegava vazio no STT (np.frombuffer de 0 bytes -> array de tamanho 0).
    return None, last_save, activation_times

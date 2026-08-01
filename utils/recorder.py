import pyaudio
import webrtcvad

from .increase_gain import gain

vad = webrtcvad.Vad(2)

def collect_speech_frames(stream: pyaudio.Stream, RATE: int, CHUNK: int):
    frames = []
    silence = 0

    while True:
        frame = stream.read(num_frames=CHUNK, exception_on_overflow=False)
        frame = gain(frame, ganho=1.5)

        if vad.is_speech(frame, RATE):
            frames.append(frame)
            silence = 0
            print("\r🎤 Falando...", end="", flush=True)

        else:
            silence += 1
            frames.append(frame)
            print("\r❌ Silêncio...", end="", flush=True)

            if silence >= 30:
                break

    print()  # pula para a próxima linha no final

    return b"".join(frames)
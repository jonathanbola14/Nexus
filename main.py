import os
import queue
import re
import threading
import time
import wave

import numpy as np
import pyaudio
import torch
from kokoro import KPipeline

from LLM import Agent
from STT import Speech_to_Text, load_stt_model
from utils.increase_gain import gain
from utils.play_file import Play
from WakeWord import WakeWord, configWakeWord

torch.set_num_threads(10)
torch.set_default_dtype(torch.int8)
player = Play()


def tts_worker():
    """Consome frases, gera os tensores de áudio (numpy) e manda direto pra fila de reprodução — sem tocar em disco."""
    while True:
        sentence = tts_queue.get()
        if sentence is STOP_SIGNAL:
            playback_queue.put(STOP_SIGNAL)
            tts_queue.task_done()
            break

        audio_chunks = []
        generator = pipeline(sentence, voice="pm_alex")
        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if audio_chunks:
            # Concatena os pedaços gerados pra essa frase em um único tensor
            full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            playback_queue.put((sentence, full_audio))

        tts_queue.task_done()


def playback_worker():
    """Toca os tensores de áudio prontos, direto da memória (sd.play), na ordem correta."""
    prebuffer = []
    stopped_early = False

    while len(prebuffer) < PREBUFFER_SIZE:
        item = playback_queue.get()
        if item is STOP_SIGNAL:
            stopped_early = True
            break
        prebuffer.append(item)

    for _sentence, audio in prebuffer:
        player.tensor(audio=audio, samplerate=SAMPLE_RATE)

    if stopped_early:
        return

    while True:
        item = playback_queue.get()
        if item is STOP_SIGNAL:
            break
        _sentence, audio = item
        player.tensor(audio, samplerate=SAMPLE_RATE)


SENTENCE_END_RE = re.compile(r'([.!?;:]+)(\s+|$)')

SAMPLE_RATE = 24000
PREBUFFER_SIZE = 2  # quantas frases sintetizadas esperar antes de começar a tocar

tts_queue = queue.Queue()
playback_queue = queue.Queue()

STOP_SIGNAL = object()

RATE = 16000
CHUNK = int(RATE * 30 / 1000)

mic = pyaudio.PyAudio()

stream = mic.open(
    format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK
)

if not os.path.exists(path="audios/noise.wav"):
    print("ruido nao coletado")
    time.sleep(0.5)
    print("fique em silencio ate voce escutar um bipe")
    ruido = stream.read(num_frames=int(RATE * 3.5), exception_on_overflow=False)
    with wave.open(f="audios/noise.wav", mode="w") as f:
        f.setframerate(framerate=16000)
        f.setnchannels(nchannels=1)
        f.setsampwidth(sampwidth=2)
        f.writeframes(data=gain(frame=ruido, ganho=2.0))
    player.file(file="audios/beep.wav")

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Inicia as threads do pipeline
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    tts_thread.start()
    playback_thread.start()

    chunks = []
    full_message = None
    reasoning_buffer = ""
    text_buffer = ""
    sentence_buffer = ""
    audio_index = 0

    pipeline = KPipeline(lang_code="p", device="cpu")

    stt_model = load_stt_model()

    wake_stream, last_save, activation_times, save_delay, cooldown, owwModel = (
        configWakeWord(mic=mic, RATE=RATE)
    )

    agent = Agent()

    config = {
        "configurable": {
            "thread_id": "nexus"
        }
    }

    os.system(command="clear")

    time.sleep(0.8)

    print("Ouvindo...")

    time.sleep(0.15)
    try:
        while True:
            audio, last_save, activation_times = WakeWord(
                wake_stream,
                RATE,
                CHUNK,
                stream,
                owwModel,
                activation_times,
                last_save,
                cooldown,
                save_delay,
            )
            if audio is None:
                continue
            input = Speech_to_Text(data=audio, RATE=RATE, model=stt_model)
            print(f"\nVoce:  {input}")

            for messages, metadata in agent.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": input
                        }
                    ]
                },
                config=config,
                stream_mode="messages"
            ):

                for block in messages.content_blocks:

                    if block["type"] == "reasoning" and block.get("reasoning"):
                        print("\n🧠 Raciocínio:\n")
                        print(block.get("reasoning", ""), end="", flush=True)
                        reasoning_buffer += block.get("reasoning", "")

                    elif block["type"] == "text" and block.get("text"):
                        print("\n\n💬 Resposta final:\n")
                        piece = block.get("text", "")
                        print(piece, end="", flush=True)

                        text_buffer += piece
                        sentence_buffer += piece

                        while True:
                            match = SENTENCE_END_RE.search(sentence_buffer)
                            if not match:
                                break
                            end_idx = match.end()
                            sentence = sentence_buffer[:end_idx].strip()
                            sentence_buffer = sentence_buffer[end_idx:]
                            if sentence:
                                tts_queue.put(sentence)

                    elif block["type"] == "tool_call" and block.get("tool_call"):
                        print(f"\n🔧Executando a ferramenta:  {block["name"]}")
                        print("\n", block["args"])
            # Frase final sem pontuação (se sobrou algo no buffer)
            if sentence_buffer.strip():
                tts_queue.put(sentence_buffer.strip())

            # Sinaliza fim do stream e espera o pipeline esvaziar
            tts_queue.put(STOP_SIGNAL)
            tts_thread.join()
            playback_thread.join()

            print()  # quebra de linha final

    except KeyboardInterrupt:
        player.stop()

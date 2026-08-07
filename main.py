import contextlib
import io
import os
import queue
import re
import threading
import time
import wave
from typing import Any

import numpy as np
import pyaudio
import torch
from kokoro import KPipeline
from langchain_core.runnables import RunnableConfig
from rich import print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from LLM import Agent
from STT import Speech_to_Text, load_stt_model
from utils.increase_gain import gain
from utils.play_file import Play
from WakeWord import WakeWord, configWakeWord

torch.set_num_threads(10)
player = Play()

console = Console()

print_lock = threading.Lock()

SENTENCE_END_RE = re.compile(r'([.!?]+)(\s+|$)')

# Abreviações comuns em pt-BR que terminam com ponto mas não fecham a frase
# (sem isso, "Dr. Silva chegou" virava duas "frases": "Dr." e "Silva chegou")
_ABBREVIATIONS = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "exmo", "exma",
    "av", "art", "pag", "pág", "cia", "ltda", "etc", "jr", "vs", "min", "seg",
}
_LAST_WORD_RE = re.compile(r'([A-Za-zÀ-ÿ]+)$')


def _is_real_sentence_end(buffer, match):
    """Filtra falsos-positivos do SENTENCE_END_RE:
    - "3." pode virar "3.14" no próximo pedaço do stream — só aceita o fim
      do buffer (sem espaço confirmado depois) quando é '!' ou '?', que
      raramente continuam; ponto sozinho espera mais texto chegar.
    - "Dr.", "etc." etc. não fecham frase — checa a palavra antes do ponto
      contra a lista de abreviações.
    """
    punct, boundary = match.group(1), match.group(2)

    if boundary == "" and "." in punct and not any(c in "!?" for c in punct):
        return False

    if "." in punct:
        word = _LAST_WORD_RE.search(buffer[:match.start()])
        if word and word.group(1).lower() in _ABBREVIATIONS:
            return False

    return True


def split_ready_sentences(buffer):
    """Tira do buffer as frases já confirmadas como completas.
    Retorna (lista_de_frases_prontas, resto_do_buffer)."""
    sentences = []
    pos = 0
    while True:
        match = SENTENCE_END_RE.search(buffer, pos)
        if not match:
            break
        if not _is_real_sentence_end(buffer, match):
            pos = match.end()
            continue
        end_idx = match.end()
        sentence = buffer[:end_idx].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end_idx:]
        pos = 0
    return sentences, buffer


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

def tts_worker():
    while True:
        sentence = tts_queue.get()
        if sentence is STOP_SIGNAL:
            playback_queue.put(STOP_SIGNAL)
            tts_queue.task_done()
            break

        audio_chunks = []
        # Silencia qualquer print/warning que o pipeline jogue no stdout/stderr.
        # ATENÇÃO: redirect_stdout troca sys.stdout pro processo inteiro, não só
        # pra esta thread — por isso precisa do print_lock, senão os prints da
        # thread principal (streaming do LLM) somem enquanto isso roda.
        with print_lock:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                generator = pipeline(sentence, voice="pm_alex")
                for gs, ps, audio in generator:
                    audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            playback_queue.put((sentence, full_audio))

        tts_queue.task_done()


def playback_worker():
    """Toca os tensores de áudio prontos, direto da memória, na ordem correta."""
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


# Run capture loop, checking for hotwords
if __name__ == "__main__":
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

    pipeline = KPipeline(lang_code="p", device="cpu")

    stt_model = load_stt_model()

    wake_stream, last_save, activation_times, save_delay, cooldown, owwModel = (
        configWakeWord(mic=mic, RATE=RATE)
    )

    agent = Agent()

    config: RunnableConfig = {
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
                player
            )
            if audio is None:
                continue

            user_input = Speech_to_Text(data=audio, RATE=RATE, model=stt_model)
            print(f"\nVoce:  {user_input}")

            # Novas threads a cada turno: as do turno anterior já terminaram
            # (elas retornam ao receberem STOP_SIGNAL e não podem ser reiniciadas)
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            playback_thread = threading.Thread(target=playback_worker, daemon=True)
            tts_thread.start()
            playback_thread.start()

            # Buffers e flags por turno: precisam ser resetados aqui, senão
            # texto/frase do turno anterior vaza (fica grudado) no próximo.
            text_buffer = ""
            sentence_buffer = ""
            printed_header_reasoning = False
            printed_header_text = False
            buffer_resposta = ""
            thinking_text = Text()
            
            try:
                with Live(console=console, refresh_per_second=60) as live:
                    for chunk in agent.stream(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ]
                        },
                        config=config,
                        stream_mode="messages",
                        version="v2",
                    ):
                        chunk: dict[str, Any]
                        token, metadata = chunk["data"]
                        for block in token.content_blocks:
                            if block["type"] == "reasoning":
                                with print_lock:
                                    if not printed_header_reasoning:
                                        print("\n🧠 Raciocínio:\n", flush=True)
                                        printed_header_reasoning = True

                                    thinking_text.append(block["reasoning"])
                                    live.update(block["reasoning"])

                            elif block["type"] == "text" and block.get("text"):
                                piece = block["text"]

                                with print_lock:
                                    if metadata["langgraph_node"] == "tools":
                                        print("\nResposta da ferramenta:  \n")
                                        print(piece, end="", flush=True)
                                        printed_header_reasoning = False
                                        printed_header_text = False
                                        buffer_resposta = ""
                                        continue
                                    else:
                                        if not printed_header_text:
                                            print("\n\n💬 Resposta final:\n", flush=True)
                                            printed_header_text = True
                                        buffer_resposta += piece
                                        live.update(Markdown(buffer_resposta, "dracula", justify="left", style="white on #323445"))

                                text_buffer += piece
                                sentence_buffer += piece

                                ready, sentence_buffer = split_ready_sentences(sentence_buffer)
                                for sentence in ready:
                                    tts_queue.put(sentence)

                            elif block["type"] == "tool_call_chunk":
                                if block["name"] != None:
                                    print(f"🔧 Chamando a ferramenta: {block['name']}")
                                elif block["args"] != '':
                                    print(block['args'], end="", flush=True)

            finally:
                # Roda mesmo se o agent.stream() acima estourar uma exceção no
                # meio do turno — sem isso, a tts_thread ficava presa pra
                # sempre esperando STOP_SIGNAL numa tts_queue que ninguém mais
                # ia alimentar, e o próximo turno criava uma 2ª thread lendo
                # da mesma fila (dois consumidores brigando pelos itens).

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
        stream.stop_stream()
        stream.close()
        wake_stream.stop_stream()
        wake_stream.close()
        mic.terminate()
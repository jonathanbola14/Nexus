import argparse
import queue
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyaudio
import torch
from kokoro import KPipeline
from langchain_core.runnables import RunnableConfig
from textual.app import App

from src.LLM import Agent
from src.STT import Speech_to_Text, load_stt_model
from src.terminal_ui import NexusApp
from src.utils.audio import resample_pcm16
from src.utils.increase_gain import gain
from src.utils.play_file import Play
from src.utils.recorder import collect_speech_frames
from src.WakeWord import WakeWord, configWakeWord

torch.set_num_threads(10)
player = Play()

pipeline = None

SENTENCE_END_RE = re.compile(r'([.!?]+)(\s+|$)')

# Abreviações comuns em pt-BR que terminam com ponto mas não fecham a frase
# (sem isso, "Dr. Silva chegou" virava duas "frases": "Dr." e "Silva chegou")
_ABBREVIATIONS = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "exmo", "exma",
    "av", "art", "pag", "pág", "cia", "ltda", "etc", "jr", "vs", "min", "seg",
}
_LAST_WORD_RE = re.compile(r'([A-Za-zÀ-ÿ]+)$')


def _cpu_times():
    with open("/proc/stat", encoding="ascii") as proc_stat:
        values = proc_stat.readline().split()[1:]
    return sum(map(int, values)), int(values[3])


def system_status(previous_cpu_times):
    """Retorna o uso aproximado de CPU e memória do sistema."""
    current_cpu_times = _cpu_times()
    total_delta = current_cpu_times[0] - previous_cpu_times[0]
    idle_delta = current_cpu_times[1] - previous_cpu_times[1]
    cpu_percent = 0 if total_delta <= 0 else (1 - idle_delta / total_delta) * 100

    memory = {}
    with open("/proc/meminfo", encoding="ascii") as proc_meminfo:
        for line in proc_meminfo:
            key, value = line.split(":", 1)
            memory[key] = int(value.split()[0])
    memory_percent = (
        (memory["MemTotal"] - memory["MemAvailable"]) / memory["MemTotal"] * 100
    )
    return current_cpu_times, f"CPU {cpu_percent:5.1f}%   MEM {memory_percent:5.1f}%"


def status_worker(app, stop_event):
    cpu_times = _cpu_times()
    while not stop_event.wait(0.3):
        cpu_times, metrics = system_status(cpu_times)
        app.publish_metrics(metrics)


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
PREBUFFER_SIZE = 1  # quantas frases sintetizadas esperar antes de começar a tocar

tts_queue = queue.Queue()
playback_queue = queue.Queue()

STOP_SIGNAL = object()

RATE = 16000
mic = pyaudio.PyAudio()
INPUT_RATE = int(mic.get_default_input_device_info()["defaultSampleRate"])
CHUNK = int(INPUT_RATE * 30 / 1000)

stream = mic.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=INPUT_RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

def tts_worker():
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


def _load_models(load_state, use_wake_word):
    """Carrega os modelos em uma thread separada e atualiza o progresso."""
    def set_progress(percent, message):
        with load_state["lock"]:
            load_state["percent"] = percent
            load_state["message"] = message

    set_progress(0, "Carregando síntese de voz")
    loaded_pipeline = KPipeline(lang_code="p", device="cpu")

    set_progress(35, "Carregando reconhecimento de voz")
    loaded_stt_model = load_stt_model()

    if use_wake_word:
        set_progress(70, "Carregando palavra de ativação")
        loaded_wake_data = configWakeWord(
            mic=mic, RATE=RATE, threshold=0.75, input_rate=INPUT_RATE
        )
    else:
        set_progress(70, "Modo sem palavra de ativação")
        loaded_wake_data = None

    set_progress(90, "Carregando agente")
    loaded_agent = Agent()

    with load_state["lock"]:
        load_state["result"] = (
            loaded_pipeline,
            loaded_stt_model,
            loaded_wake_data,
            loaded_agent,
        )
        load_state["percent"] = 100
        load_state["message"] = "Modelos carregados"


def load_models(load_state, use_wake_word):
    try:
        _load_models(load_state, use_wake_word)
    except Exception as error:
        with load_state["lock"]:
            load_state["error"] = error
            load_state["message"] = "Falha ao carregar modelos"


PROJECT_ROOT = Path(__file__).resolve().parent

def run_assistant(app: App, use_wake_word: bool):
    global pipeline
    load_state = {
        "lock": threading.Lock(),
        "percent": 0,
        "message": "Iniciando",
        "result": None,
        "error": None,
    }

    noise_path = PROJECT_ROOT / "audios" / "noise.wav"
    beep_path = PROJECT_ROOT / "audios" / "beep.wav"
    status_stop_event = threading.Event()
    status_thread = threading.Thread(
        target=status_worker,
        args=(app, status_stop_event),
        daemon=True,
    )
    loader_thread = threading.Thread(
        target=load_models,
        args=(load_state, use_wake_word),
    )
    wake_stream = None
    close_app = True

    def stop_audio():
        player.stop()
        stream.stop_stream()

    app.shutdown_callback = stop_audio
    status_thread.start()

    try:
        if not noise_path.exists():
            app.set_status("Fique em silêncio para calibrar o microfone", "INICIALIZAÇÃO")
            time.sleep(0.5)
            ruido = resample_pcm16(
                stream.read(
                    num_frames=int(INPUT_RATE * 3.5),
                    exception_on_overflow=False,
                ),
                INPUT_RATE,
                RATE,
            )
            with wave.open(str(noise_path), mode="w") as noise_file:
                noise_file.setframerate(16000)
                noise_file.setnchannels(1)
                noise_file.setsampwidth(2)
                noise_file.writeframes(data=gain(frame=ruido, ganho=2.0))
            player.file(file=str(beep_path))

        config: RunnableConfig = {"configurable": {"thread_id": "nexus"}}
        loader_thread.start()
        while loader_thread.is_alive():
            with load_state["lock"]:
                percent = load_state["percent"]
                message = load_state["message"]
            app.publish_progress(percent, message)
            time.sleep(0.1)
        loader_thread.join()

        if load_state["error"] is not None:
            app.set_status(f"Falha ao iniciar: {load_state['error']}", "ERRO")
            close_app = False
            return

        pipeline, stt_model, wake_data, agent = load_state["result"]
        app.clear_progress()
        if use_wake_word:
            wake_stream, last_save, activation_times, save_delay, cooldown, owwModel = wake_data
        else:
            last_save = activation_times = save_delay = cooldown = owwModel = None

        session_history = ""
        app.publish_session("Aguardando a palavra de ativação.", "OUVINDO")

        while not app.stop_requested.is_set():
            if use_wake_word:
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
                    player,
                    app,
                    input_rate=INPUT_RATE,
                )
            else:
                audio = collect_speech_frames(
                    stream, RATE, CHUNK, app, input_rate=INPUT_RATE
                )

            if not audio or app.stop_requested.is_set():
                continue

            user_input = Speech_to_Text(data=audio, RATE=RATE, model=stt_model)
            if user_input in (
                "Desligar.", "Desligar", "Encerrar", "Encerrar.", "Desliga.", "Desliga"
            ):
                app._shutdown()

            session_history += f"Você\n{user_input}\n\n"
            app.publish_session(session_history, "PENSANDO")

            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            playback_thread = threading.Thread(target=playback_worker, daemon=True)
            tts_thread.start()
            playback_thread.start()

            sentence_buffer = ""
            buffer_resposta = ""
            buffer_raciocinio = ""
            cpu_times = _cpu_times()

            try:
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                    stream_mode="messages",
                    version="v2",
                ):
                    chunk: dict[str, Any]
                    token, metadata = chunk["data"]
                    cpu_times, metrics = system_status(cpu_times)
                    app.publish_metrics(metrics)

                    for block in token.content_blocks:
                        if block["type"] == "reasoning":
                            piece = block.get("reasoning") or block.get("text", "")
                            if piece:
                                buffer_raciocinio += piece
                                transcript = session_history + "Nexus\n"
                                app.publish_session(
                                    transcript,
                                    "RACIOCINANDO",
                                    response=buffer_resposta,
                                    reasoning=buffer_raciocinio,
                                )
                            continue

                        if block["type"] == "text" and block.get("text"):
                            piece = block["text"]
                            if metadata["langgraph_node"] == "tools":
                                buffer_resposta = ""
                                continue

                            buffer_resposta += piece
                            app.publish_session(
                                session_history + "Nexus\n",
                                "RESPONDENDO",
                                response=buffer_resposta,
                                reasoning=buffer_raciocinio,
                            )
                            sentence_buffer += piece
                            ready, sentence_buffer = split_ready_sentences(sentence_buffer)
                            for sentence in ready:
                                tts_queue.put(sentence)

            except Exception:
                fallback = (
                    "Não consegui acessar o serviço de inteligência agora. "
                    "Verifique a chave da API e a conexão com a internet e tente novamente."
                )
                buffer_resposta += ("\n\n" if buffer_resposta else "") + fallback
                sentence_buffer += " " + fallback
                app.publish_session(
                    session_history + "Nexus\n",
                    "ERRO",
                    response=buffer_resposta,
                    reasoning=buffer_raciocinio,
                )

            finally:
                if sentence_buffer.strip():
                    tts_queue.put(sentence_buffer.strip())
                tts_queue.put(STOP_SIGNAL)
                tts_thread.join()
                playback_thread.join()

                if buffer_resposta.strip():
                    session_history += f"Nexus\n{buffer_resposta.strip()}\n\n"
                app.publish_session(session_history, "OUVINDO")

    except KeyboardInterrupt:
        app.stop_requested.set()
    except OSError as error:
        if app.stop_requested.is_set():
            close_app = True
        else:
            app.set_status(f"Falha de áudio: {error}", "ERRO")
            close_app = False
    except Exception as error:
        app.set_status(f"Falha no assistente: {error}", "ERRO")
        close_app = False
    finally:
        status_stop_event.set()
        status_thread.join()
        player.stop()
        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            pass
        if wake_stream is not None:
            try:
                wake_stream.stop_stream()
                wake_stream.close()
            except OSError:
                pass
        mic.terminate()
        if close_app:
            app.finish()


def main():
    parser = argparse.ArgumentParser(description="Assistente de voz Nexus")
    parser.add_argument(
        "--sem-ativacao",
        action="store_true",
        help="Captura a fala diretamente, sem exigir a palavra de ativação",
    )
    args = parser.parse_args()
    use_wake_word = not args.sem_ativacao
    NexusApp(runner=lambda app: run_assistant(app, use_wake_word)).run()


if __name__ == "__main__":
    main()
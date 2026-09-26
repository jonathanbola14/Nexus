import contextlib
import ctypes
import os
import re
import threading
from typing import Any

import numpy as np
import pyaudio
from kokoro import KPipeline
from textual.app import App
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, RichLog, Static
from textual.widgets._header import Header

from LLM import Agent
from STT import Speech_to_Text, load_stt_model
from utils.play_file import Play
from WakeWord import WakeWord, configWakeWord

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.c_int, ctypes.c_char_p
)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except OSError:
    pass

SAMPLE_RATE = 24000
RATE = 16000
CHUNK = int(RATE * 30 / 1000)


class MainScreen(Screen):
    CSS_PATH = "css/style.tcss"
    TITLE = "Nexus"

    def compose(self):
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                Static("NEXUS", id="brand"),
                Static("ASSISTENTE DE VOZ", classes="eyebrow"),
                Static("CARREGANDO MODELOS", id="assistant-status"),
                Static("SISTEMA", classes="section-label"),
                Static("Microfone\nEntrada de áudio", classes="system-item"),
                Static("Núcleo\nAgente inteligente", classes="system-item"),
                Static("Voz\nSíntese em tempo real", classes="system-item"),
                id="sidebar-content",
            ),
            id="sidebar",
        )
        self.button = Button("Iniciar conversa", id="start-button")
        self.button.disabled = True
        yield Vertical(
            Container(
                Static("CONVERSA", classes="eyebrow"),
                Static("Assistente pessoal", id="conversation-title"),
                id="conversation-heading",
            ),
            RichLog(id="conversation-log", wrap=True, markup=False),
            Container(
                Static("NEXUS  /  ASSISTENTE DE VOZ", id="input-hint"),
                self.button,
                id="container-button",
            ),
            id="conversa",
        )

    def on_mount(self) -> None:
        self._shutdown = threading.Event()
        self._start = threading.Event()
        self._ready = False
        self.query_one("#conversation-log", RichLog).write(
            "Nexus\nOlá. Estou pronto para ajudar."
        )
        self._worker = threading.Thread(target=self._load_and_run, daemon=True)
        self._worker.start()

    def on_button_pressed(self, event: Button.Pressed):
        if not self._ready:
            return
        if self._start.is_set():
            self._start.clear()
            self.button.label = "Iniciar conversa"
            self._set_status("CONVERSA PAUSADA")
        else:
            self._start.set()
            self.button.label = "Parar conversa"
            self._set_status("OUVINDO PALAVRA DE ATIVAÇÃO")

    def _set_status(self, status: str) -> None:
        self.query_one("#assistant-status", Static).update(status)

    def _append_message(self, speaker: str, text: str) -> None:
        self.query_one("#conversation-log", RichLog).write(
            f"{speaker}\n{text.strip()}"
        )

    def _load_and_run(self) -> None:
        mic = None
        wake_stream = None
        input_stream = None
        player = None
        try:
            self.app.call_from_thread(self._set_status, "CARREGANDO SÍNTESE DE VOZ")
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                pipeline = KPipeline(lang_code="p", device="cpu")

            self.app.call_from_thread(self._set_status, "CARREGANDO RECONHECIMENTO DE VOZ")
            stt_model = load_stt_model()

            self.app.call_from_thread(self._set_status, "INICIANDO ÁUDIO E AGENTE")
            mic = pyaudio.PyAudio()
            wake_data = configWakeWord(mic=mic, RATE=RATE)
            wake_stream, last_save, activation_times, save_delay, cooldown, wake_model = wake_data
            input_stream = mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            player = Play()
            agent = Agent()
            self.app.call_from_thread(self._enable_conversation)

            config = {"configurable": {"thread_id": "nexus"}}
            wake_status = _WakeStatus(self)
            while not self._shutdown.is_set():
                if not self._start.wait(timeout=0.1):
                    continue

                audio, last_save, activation_times = WakeWord(
                    wake_stream,
                    RATE,
                    CHUNK,
                    input_stream,
                    wake_model,
                    activation_times,
                    last_save,
                    cooldown,
                    save_delay,
                    player,
                    wake_status,
                )
                if audio is None or not self._start.is_set():
                    continue

                self.app.call_from_thread(self._set_status, "TRANSCRIBINDO")
                user_text = Speech_to_Text(data=audio, RATE=RATE, model=stt_model).strip()
                if not user_text:
                    self.app.call_from_thread(self._set_status, "OUVINDO PALAVRA DE ATIVAÇÃO")
                    continue

                self.app.call_from_thread(self._append_message, "Você", user_text)
                self.app.call_from_thread(self._set_status, "PENSANDO")
                answer = ""
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config=config,
                    stream_mode="messages",
                    version="v2",
                ):
                    if not self._start.is_set():
                        break
                    token, metadata = chunk["data"]
                    if metadata.get("langgraph_node") == "tools":
                        continue
                    for block in token.content_blocks:
                        if block.get("type") == "text" and block.get("text"):
                            answer += block["text"]
                if answer and self._start.is_set():
                    self.app.call_from_thread(self._append_message, "Nexus", answer)
                    self.app.call_from_thread(self._set_status, "RESPONDENDO")
                    self._speak(pipeline, player, answer)

                if self._start.is_set():
                    self.app.call_from_thread(self._set_status, "OUVINDO PALAVRA DE ATIVAÇÃO")
        except Exception as error:  # noqa: BLE001
            self.app.call_from_thread(self._show_error, str(error))
        finally:
            for audio_stream in (wake_stream, input_stream):
                if audio_stream is not None:
                    try:
                        audio_stream.stop_stream()
                        audio_stream.close()
                    except OSError:
                        pass
            if player is not None:
                player.stop()
            if mic is not None:
                mic.terminate()

    def _enable_conversation(self) -> None:
        self._ready = True
        self.button.disabled = False
        self._set_status("PRONTO PARA CONVERSAR")

    def _show_error(self, message: str) -> None:
        self._set_status("FALHA AO INICIAR")
        self.button.label = "Indisponível"
        self.button.disabled = True
        self.query_one("#conversation-log", RichLog).write(f"Erro\n{message}")

    @staticmethod
    def _speak(pipeline: KPipeline, player: Play, text: str) -> None:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        for sentence in sentences:
            audio_chunks = [audio for _, _, audio in pipeline(sentence + ".", voice="pm_alex")]
            if audio_chunks:
                audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
                player.tensor(audio, samplerate=SAMPLE_RATE)

    def on_unmount(self) -> None:
        self._shutdown.set()
        self._start.set()


class _WakeStatus:
    def __init__(self, screen: MainScreen) -> None:
        self.screen = screen

    def update(self, _renderable: Any) -> None:
        self.screen.app.call_from_thread(
            self.screen._set_status, "PALAVRA DE ATIVAÇÃO DETECTADA"
        )


class Nexus(App):
    def compose(self):
        yield Static()

    def on_mount(self):
        self.push_screen(MainScreen())


app = Nexus()
app.run()

import threading
from collections.abc import Callable
from typing import ClassVar

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Button, Footer, Header, Static


class NexusApp(App):
    _is_textual_bridge = True
    TITLE = "NEXUS"
    SUB_TITLE = "Assistente pessoal de voz"
    CSS = """
    Screen {
        background: #111918;
        color: #e8eee9;
    }

    Header {
        background: #172321;
        color: #b8f28b;
    }

    #workspace {
        height: 1fr;
        padding: 1 2;
    }

    #conversation {
        width: 1fr;
        height: 1fr;
        border: round #52715a;
        padding: 1 2;
        margin-right: 1;
    }

    #sidebar {
        width: 32;
        height: 1fr;
        border: round #415752;
        padding: 1 2;
    }

    .section-title {
        color: #b8f28b;
        text-style: bold;
        height: 2;
    }

    #transcript {
        height: 1fr;
        padding-top: 1;
    }

    #state {
        color: #b8f28b;
        text-style: bold;
        height: 2;
    }

    #detail {
        height: 1fr;
        color: #bdc9c2;
    }

    #controls {
        height: 3;
        align: center middle;
    }

    Button {
        margin: 0 1;
        border: none;
        background: #284238;
        color: #f0f5f0;
    }

    Button:hover {
        background: #3d6049;
    }

    Footer {
        background: #172321;
    }

    Screen.compact #workspace {
        layout: vertical;
        padding: 0;
    }

    Screen.compact #conversation {
        height: 2fr;
        margin-right: 0;
        margin-bottom: 1;
        padding: 1;
    }

    Screen.compact #sidebar {
        width: 1fr;
        height: 1fr;
        padding: 1;
    }
    """
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("space", "next_step", "Próxima etapa"),
        ("r", "restart", "Reiniciar"),
        ("q", "request_exit", "Sair"),
    ]

    STEPS: ClassVar[list[tuple[str, str, str]]] = [
        ("OUVINDO", "Aguardando a palavra de ativação.\n\nDiga: Nexus.", "Microfone\nPronto para ouvir"),
        ("PENSANDO", "Você\nQue horas são?", "Entrada\nFala reconhecida"),
        ("RACIOCINANDO", "Você\nQue horas são?\n\nNexus\nConsultando o horário atual...", "Ferramenta\nConsulta de hora"),
        ("RESPONDENDO", "Você\nQue horas são?\n\nNexus\nAgora são 14 horas e 32 minutos.", "Voz\nAlex · pt-BR\n\nSistema\nCPU e memória locais"),
        ("OUVINDO", "Você\nQue horas são?\n\nNexus\nAgora são 14 horas e 32 minutos.\n\nAguardando a próxima solicitação.", "Sessão\nPronta"),
    ]

    def __init__(self, runner: Callable[["NexusApp"], None] | None = None) -> None:
        super().__init__()
        self.runner = runner
        self.stop_requested = threading.Event()
        self.shutdown_callback = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with Vertical(id="conversation"):
                yield Static("CONVERSA", classes="section-title")
                yield Static("", id="transcript", markup=False)
            with Vertical(id="sidebar"):
                yield Static("ESTADO DO ASSISTENTE", classes="section-title")
                yield Static("", id="state", markup=False)
                yield Static("", id="detail", markup=False)
                yield Static("", id="progress", markup=False)
                yield Static("CPU --   MEM --", id="metrics", markup=False)
        with Horizontal(id="controls"):
            if getattr(self, "runner", None) is None:
                yield Button("Avançar", id="next", variant="success")
                yield Button("Reiniciar", id="restart")
            else:
                yield Button("Encerrar", id="quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.step_index = 0
        if self.runner is not None:
            self.query_one("#state", Static).update("INICIALIZANDO")
            self.query_one("#transcript", Static).update("Preparando o assistente de voz...")
            self.query_one("#detail", Static).update("Aguardando carregamento dos modelos")
            self.run_worker(lambda: self.runner(self), thread=True, exclusive=True)
            return

        self.show_step()
        self.demo_timer = self.set_interval(2.2, self.action_next_step)

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 60, "compact")

    def show_step(self) -> None:
        state, transcript, detail = self.STEPS[self.step_index]
        self.query_one("#state", Static).update(state)
        self.query_one("#transcript", Static).update(transcript)
        self.query_one("#detail", Static).update(detail)

    def set_status(self, message: str, state: str | None = None) -> None:
        self.call_from_thread(self._set_status, message, state)

    def _set_status(self, message: str, state: str | None = None) -> None:
        self.query_one("#detail", Static).update(message)
        if state:
            self.query_one("#state", Static).update(state)

    def publish_progress(self, percent: int, message: str) -> None:
        value = f"{message}\n[{'#' * (percent // 5)}{'-' * (20 - percent // 5)}] {percent:3d}%"
        self.call_from_thread(self._publish_value, "#progress", value)

    def clear_progress(self) -> None:
        self.call_from_thread(self._publish_value, "#progress", "")

    def publish_metrics(self, message: str) -> None:
        self.call_from_thread(self._publish_value, "#metrics", message)

    def _publish_value(self, selector: str, value: str) -> None:
        self.query_one(selector, Static).update(value)

    def publish_session(
        self,
        transcript: str,
        state: str,
        response: str = "",
        reasoning: str = "",
    ) -> None:
        content = transcript
        if reasoning:
            content += f"\nRACIOCÍNIO\n{reasoning}"
        if response:
            content += f"\n{response}"
        self.call_from_thread(self._publish_session, content, state)

    def _publish_session(self, content: str, state: str) -> None:
        state_details = {
            "OUVINDO": "Aguardando a palavra de ativação ou uma fala.",
            "CAPTURANDO": "Capturando sua fala pelo microfone...",
            "PENSANDO": "Aguardando resposta do agente NVIDIA...",
            "RACIOCINANDO": "Processando a solicitação...",
            "RESPONDENDO": "Resposta recebida; sintetizando a fala...",
            "ERRO": "Ocorreu uma falha durante esta etapa.",
        }
        self.query_one("#transcript", Static).update(content)
        self.query_one("#state", Static).update(state)
        self.query_one("#detail", Static).update(state_details.get(state, ""))

    def finish(self) -> None:
        self.call_from_thread(self.exit)

    def action_next_step(self) -> None:
        if self.runner is not None:
            return
        self.step_index = (self.step_index + 1) % len(self.STEPS)
        self.show_step()

    def action_restart(self) -> None:
        if self.runner is not None:
            return
        self.step_index = 0
        self.show_step()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self.action_next_step()
        elif event.button.id == "restart":
            self.action_restart()
        elif event.button.id == "quit":
            self.action_request_exit()

    def action_request_exit(self) -> None:
        self.stop_requested.set()
        try:
            if self.shutdown_callback is not None:
                self.shutdown_callback()
        finally:
            self.exit()


class NexusPreview(NexusApp):
    runner = None

def main() -> None:
    NexusPreview().run()


if __name__ == "__main__":
    main()

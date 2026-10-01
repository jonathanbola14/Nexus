import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Footer,
    Header,
    LoadingIndicator,
    ProgressBar,
    Select,
    Static,
)


def format_transcript(transcript: str) -> Text:
    formatted = Text()
    speaker_styles = {
        "Você": "bold #79d8c2",
        "Nexus": "bold #c4ef91",
    }
    for line in transcript.splitlines(keepends=True):
        speaker = line.rstrip("\r\n")
        style = speaker_styles.get(speaker)
        if style:
            formatted.append(speaker, style=style)
            formatted.append(line[len(speaker) :])
        else:
            formatted.append(line)
    return formatted


APP_THEMES = (
    Theme(
        name="nexus",
        primary="#c4ef91",
        secondary="#79d8c2",
        accent="#f0cd83",
        foreground="#e4ece6",
        background="#101715",
        surface="#192622",
        panel="#151f1c",
        success="#79d8a0",
        warning="#f0cd83",
        error="#ef8c83",
        dark=True,
    ),
    Theme(
        name="ocean",
        primary="#91d9e3",
        secondary="#79c3d2",
        accent="#9bdce4",
        foreground="#e0f1f2",
        background="#10191c",
        surface="#17272b",
        panel="#152126",
        success="#86d2b0",
        warning="#e9c579",
        error="#ef9288",
        dark=True,
    ),
    Theme(
        name="amber",
        primary="#efc779",
        secondary="#d5ae71",
        accent="#efc779",
        foreground="#f2e5ca",
        background="#191712",
        surface="#29231a",
        panel="#211d16",
        success="#b8d28b",
        warning="#efc779",
        error="#ed8a73",
        dark=True,
    ),
    Theme(
        name="paper",
        primary="#426b4e",
        secondary="#597864",
        accent="#765820",
        foreground="#303b34",
        background="#f1f3ed",
        surface="#e2e8df",
        panel="#e9eee6",
        success="#426b4e",
        warning="#765820",
        error="#9a4540",
        dark=False,
    ),
)

THEME_CHOICES = (
    ("Verde Nexus", "nexus"),
    ("Azul oceano", "ocean"),
    ("Âmbar", "amber"),
    ("Claro", "paper"),
    ("")
)

STYLE_CHOICES = (
    ("Minimalista", "minimal"),
    ("Com painéis", "panels"),
)

STATE_DETAILS = {
    "OUVINDO": "Aguardando a palavra de ativação ou uma fala.",
    "CAPTURANDO": "Capturando sua fala pelo microfone...",
    "PENSANDO": "Aguardando resposta do agente NVIDIA...",
    "RACIOCINANDO": "Processando a solicitação...",
    "RESPONDENDO": "Resposta recebida; sintetizando a fala...",
    "ERRO": "Ocorreu uma falha durante esta etapa.",
}

ACTIVITY_STATES = {
    "INICIALIZANDO",
    "CAPTURANDO",
    "PENSANDO",
    "RACIOCINANDO",
    "RESPONDENDO",
}


class SettingsScreen(ModalScreen[tuple[str, str] | None]):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "cancel", "Cancelar"),
    ]
    def __init__(self, theme: str, style: str) -> None:
        super().__init__()
        self.current_theme = theme
        self.current_style = style

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 92, "compact")

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Static("APARÊNCIA", classes="settings-title")
            yield Static("Tema", classes="settings-label")
            yield Select(
                list(THEME_CHOICES),
                value=self.current_theme,
                allow_blank=False,
                id="theme-select",
            )
            yield Static("Estilo", classes="settings-label")
            yield Select(
                list(STYLE_CHOICES),
                value=self.current_style,
                allow_blank=False,
                id="style-select",
            )
            with Horizontal(id="settings-actions"):
                yield Button("Cancelar", id="settings-cancel")
                yield Button("Aplicar", id="settings-apply", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-cancel":
            self.dismiss(None)
        elif event.button.id == "settings-apply":
            theme = self.query_one("#theme-select", Select).value
            style = self.query_one("#style-select", Select).value
            self.dismiss((str(theme), str(style)))

    def action_cancel(self) -> None:
        self.dismiss(None)


class NexusApp(App):
    _is_textual_bridge = True
    TITLE = "NEXUS"
    SUB_TITLE = "Assistente pessoal de voz"
    assistant_state: reactive[str] = reactive("OUVINDO")
    CSS_PATH = str(Path(__file__, "css").with_name("terminal_ui.tcss"))
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("space", "next_step", "Próxima etapa"),
        ("r", "restart", "Reiniciar"),
        ("t", "open_settings", "Aparência"),
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
        self.preference_theme = "nexus"
        self.preference_style = "minimal"
        self.stop_requested = threading.Event()
        self.shutdown_callback = None
        self._published_session = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="workspace"):
            with Vertical(id="conversation"):
                yield Static("CONVERSA", classes="section-title")
                yield Static(
                    "FERRAMENTA  --", id="tool", classes="tool-indicator"
                )
                with VerticalScroll(id="transcript"):
                    with Vertical(classes="transcript-section", id="history-section"):
                        yield Static("", id="history", classes="transcript-content", markup=False)
                    with Vertical(classes="transcript-section", id="reasoning-section"):
                        yield Static("RACIOCÍNIO", classes="transcript-label")
                        yield Static("", id="reasoning", classes="transcript-content", markup=False)
                    with Vertical(classes="transcript-section", id="response-section"):
                        yield Static("RESPOSTA", classes="transcript-label")
                        yield Static("", id="response", classes="transcript-content", markup=False)
            with Horizontal(id="statusbar"):
                yield LoadingIndicator(id="activity-indicator")
                yield Static("", id="state", markup=False)
                yield Static("", id="detail", markup=False)
                yield Static("CPU --   MEM --", id="metrics", markup=False)
            with Vertical(id="progress-wrap"):
                yield Static("", id="progress", markup=False)
                yield ProgressBar(total=100, show_eta=False, id="progress-bar")
        with Horizontal(id="controls"):
            if getattr(self, "runner", None) is None:
                yield Button("Avançar", id="next", variant="success")
                yield Button("Reiniciar", id="restart")
                yield Button("Aparência", id="settings-button")
            else:
                yield Button("Aparência", id="settings-button")
                yield Button("Encerrar", id="quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        for theme in APP_THEMES:
            self.register_theme(theme)
        self.theme = self.preference_theme
        self.step_index = 0
        if self.runner is not None:
            self.assistant_state = "INICIALIZANDO"
            self.query_one("#history", Static).update("Preparando o assistente de voz...")
            self.query_one("#detail", Static).update("Aguardando carregamento dos modelos")
            self.run_worker(lambda: self.runner(self), thread=True, exclusive=True)
            return

        self.show_step()
        self.demo_timer = self.set_interval(2.2, self.action_next_step)

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 92, "compact")

    def show_step(self) -> None:
        state, transcript, detail = self.STEPS[self.step_index]
        self.assistant_state = state
        self.query_one("#history", Static).update(format_transcript(transcript))
        self.query_one("#detail", Static).update(detail)
        tool = "Hour" if self.step_index in {2, 3, 4} else "Aguardando chamada"
        self.query_one("#tool", Static).update(f"FERRAMENTA  {tool}")

    def set_status(self, message: str, state: str | None = None) -> None:
        self.call_from_thread(self._set_status, message, state)

    def _set_status(self, message: str, state: str | None = None) -> None:
        self._update_static("#detail", message)
        if state:
            self.assistant_state = state

    def _update_static(self, selector: str, value: str) -> None:
        self.query_one(selector, Static).update(value)

    def publish_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, percent))
        self.call_from_thread(self._set_progress, percent, message)

    def clear_progress(self) -> None:
        self.call_from_thread(self._clear_progress)

    def _set_progress(self, percent: int, message: str) -> None:
        self._update_static("#progress", message)
        self.query_one("#progress-bar", ProgressBar).update(progress=percent)
        self.query_one("#progress-wrap").display = True

    def _clear_progress(self) -> None:
        self._update_static("#progress", "")
        self.query_one("#progress-bar", ProgressBar).update(progress=0)
        self.query_one("#progress-wrap").display = False

    def publish_metrics(self, message: str) -> None:
        self.call_from_thread(self._publish_value, "#metrics", message)

    def publish_tool(self, name: str) -> None:
        self.call_from_thread(self._publish_value, "#tool", f"FERRAMENTA  {name}")

    def _publish_value(self, selector: str, value: str) -> None:
        self._update_static(selector, value)

    def publish_session(
        self,
        transcript: str,
        state: str,
        response: str = "",
        reasoning: str = "",
    ) -> None:
        self.call_from_thread(
            self._publish_session, transcript, state, response, reasoning
        )

    def _publish_session(
        self, transcript: str, state: str, response: str = "", reasoning: str = ""
    ) -> None:
        previous = self._published_session
        has_new_content = previous is None or (transcript, reasoning, response) != previous
        scroll = self.query_one("#transcript", VerticalScroll)
        follow_output = scroll.scroll_y >= scroll.max_scroll_y - 1

        if previous is None or transcript != previous[0]:
            self._update_static("#history", format_transcript(transcript))
            if state == "PENSANDO":
                self._update_static("#tool", "FERRAMENTA  aguardando")
        if previous is None or reasoning != previous[1]:
            self._update_static("#reasoning", reasoning)
        if previous is None or response != previous[2]:
            self._update_static("#response", response)
        self._published_session = (transcript, reasoning, response)
        self.query_one("#reasoning-section").display = bool(reasoning)
        self.query_one("#response-section").display = bool(response)
        self.assistant_state = state
        self._update_static("#detail", STATE_DETAILS.get(state, ""))
        if has_new_content and follow_output:
            scroll.scroll_end(animate=False)

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
        elif event.button.id == "settings-button":
            self.action_open_settings()
        elif event.button.id == "quit":
            self.action_request_exit()

    def watch_assistant_state(self, state: str) -> None:
        self._update_static("#state", state)
        self.query_one("#activity-indicator", LoadingIndicator).display = (
            state in ACTIVITY_STATES
        )

    def _apply_display_style(self, theme: str, style: str) -> None:
        self.preference_theme = theme
        self.preference_style = style
        self.theme = self.preference_theme
        screen = self.screen
        screen.remove_class("style-minimal", "style-panels")
        screen.add_class(f"style-{self.preference_style}")

    def action_open_settings(self) -> None:
        self.push_screen(
            SettingsScreen(self.preference_theme, self.preference_style),
            self._apply_settings,
        )

    def _apply_settings(self, settings: tuple[str, str] | None) -> None:
        if settings is None:
            return
        self._apply_display_style(*settings)

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

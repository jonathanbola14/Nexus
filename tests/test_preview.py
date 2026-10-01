import asyncio
import threading

from textual.widgets import Footer, Header, LoadingIndicator, ProgressBar, Select

from src.terminal_ui import NexusApp, NexusPreview, SettingsScreen, format_transcript


def test_transcript_styles_speakers_without_changing_content():
    transcript = "Você\nOi.\n\nNexus\nOlá.\n"

    formatted = format_transcript(transcript)

    assert formatted.plain == transcript
    assert len(formatted.spans) == 2
    assert all("bold" in span.style for span in formatted.spans)


def test_preview_states_and_controls():
    async def run_app():
        async with NexusPreview().run_test() as pilot:
            state = pilot.app.query_one("#state")
            assert str(state.render()) == "OUVINDO"
            assert pilot.app.query_one(LoadingIndicator).display is False

            await pilot.click("#next")
            assert str(state.render()) == "PENSANDO"
            assert pilot.app.query_one(LoadingIndicator).display is True

            await pilot.click("#restart")
            assert str(state.render()) == "OUVINDO"
            assert pilot.app.query_one(LoadingIndicator).display is False

    asyncio.run(run_app())


def test_preview_uses_compact_layout_on_narrow_terminal():
    async def run_app():
        async with NexusPreview().run_test(size=(90, 20)) as pilot:
            await pilot.pause()
            assert "compact" in pilot.app.screen.classes

    asyncio.run(run_app())


def test_header_and_footer_stay_docked_around_flexible_workspace():
    async def run_app():
        async with NexusPreview().run_test(size=(100, 32)) as pilot:
            header = pilot.app.query_one(Header)
            footer = pilot.app.query_one(Footer)
            workspace = pilot.app.query_one("#workspace")

            assert header.region.y == 0
            assert footer.region.bottom == pilot.app.screen.size.height
            assert workspace.region.height > 1

    asyncio.run(run_app())


def test_settings_screen_applies_theme_and_interface_style():
    async def run_app():
        async with NexusPreview().run_test(size=(100, 24)) as pilot:
            await pilot.click("#settings-button")
            assert isinstance(pilot.app.screen, SettingsScreen)

            pilot.app.screen.query_one("#theme-select", Select).value = "ocean"
            pilot.app.screen.query_one("#style-select", Select).value = "panels"
            await pilot.click("#settings-apply")

            assert pilot.app.preference_theme == "ocean"
            assert pilot.app.preference_style == "panels"
            assert pilot.app.theme == "ocean"
            assert "style-panels" in pilot.app.screen.classes

    asyncio.run(run_app())


def test_settings_screen_fits_compact_terminal():
    async def run_app():
        async with NexusPreview().run_test(size=(50, 24)) as pilot:
            await pilot.click("#settings-button")
            dialog = pilot.app.screen.query_one("#settings-dialog")

            assert "compact" in pilot.app.screen.classes
            assert dialog.region.width <= pilot.app.screen.size.width

    asyncio.run(run_app())


def test_progress_bar_is_clamped_and_cleared_from_worker():
    published = threading.Event()

    def runner(app):
        app.publish_progress(140, "Carregando modelos")
        published.set()

    async def run_app():
        async with NexusApp(runner=runner).run_test() as pilot:
            await pilot.pause(0.1)
            assert published.is_set()
            assert str(pilot.app.query_one("#progress").render()) == "Carregando modelos"
            assert pilot.app.query_one(ProgressBar).progress == 100
            assert pilot.app.query_one("#progress-wrap").display is True

            pilot.app._clear_progress()
            assert pilot.app.query_one(ProgressBar).progress == 0
            assert pilot.app.query_one("#progress-wrap").display is False

    asyncio.run(run_app())


def test_assistant_worker_can_publish_to_textual_widgets():
    published = threading.Event()

    def runner(app):
        app.publish_session(
            "Você\nHola.", "PENSANDO", response="Resposta", reasoning="Raciocínio"
        )
        app.publish_progress(20, "Carregando [modelos]")
        app.clear_progress()
        app.publish_tool("Hour")
        published.set()

    async def run_app():
        async with NexusApp(runner=runner).run_test() as pilot:
            await pilot.pause(0.1)
            assert published.is_set()
            assert "Hola." in str(pilot.app.query_one("#history").render())
            assert str(pilot.app.query_one("#reasoning").render()) == "Raciocínio"
            assert str(pilot.app.query_one("#response").render()) == "Resposta"
            assert pilot.app.query_one("#transcript").can_focus is True
            assert str(pilot.app.query_one("#state").render()) == "PENSANDO"
            assert str(pilot.app.query_one("#tool").render()) == "FERRAMENTA  Hour"
            assert pilot.app.query_one("#tool").parent.id == "conversation"
            assert len(pilot.app.query_one("#statusbar").query("#tool")) == 0
            section_titles = [
                str(title.render()) for title in pilot.app.query(".section-title")
            ]
            assert section_titles == ["CONVERSA"]
            assert "Aguardando resposta do agente" in str(
                pilot.app.query_one("#detail").render()
            )
            assert pilot.app.query_one("#statusbar") is not None
            assert str(pilot.app.query_one("#progress").render()).strip() == ""

    asyncio.run(run_app())

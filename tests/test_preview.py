import asyncio
import threading

from preview import NexusApp, NexusPreview


def test_preview_states_and_controls():
    async def run_app():
        async with NexusPreview().run_test() as pilot:
            state = pilot.app.query_one("#state")
            assert str(state.render()) == "OUVINDO"

            await pilot.click("#next")
            assert str(state.render()) == "PENSANDO"

            await pilot.click("#restart")
            assert str(state.render()) == "OUVINDO"

    asyncio.run(run_app())


def test_preview_uses_compact_layout_on_narrow_terminal():
    async def run_app():
        async with NexusPreview().run_test(size=(50, 20)) as pilot:
            await pilot.pause()
            assert "compact" in pilot.app.screen.classes

    asyncio.run(run_app())


def test_assistant_worker_can_publish_to_textual_widgets():
    published = threading.Event()

    def runner(app):
        app.publish_session("Você\nHola.", "PENSANDO")
        app.publish_progress(20, "Carregando [modelos]")
        app.clear_progress()
        published.set()

    async def run_app():
        async with NexusApp(runner=runner).run_test() as pilot:
            await pilot.pause(0.1)
            assert published.is_set()
            assert "Hola." in str(pilot.app.query_one("#transcript").render())
            assert str(pilot.app.query_one("#state").render()) == "PENSANDO"
            assert "Aguardando resposta do agente" in str(
                pilot.app.query_one("#detail").render()
            )
            assert str(pilot.app.query_one("#progress").render()).strip() == ""

    asyncio.run(run_app())

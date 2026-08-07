from textual.app import App
from textual.widgets import Label, Input, Header
from textual.containers import Vertical, Container

class Nexus(App):
    CSS_PATH = "css/style.tcss"

    def compose(self):
        yield Header(True)
        yield Vertical(
            id="container"
        )
        
        yield Container(
            Vertical(id="sidebar"),
            Input(id="input")
        )

    def on_mount(self):
        container = self.query_one("#container")

        for i in range(10):
            novo_widget = Label(f"widget numero {i}")
            container.mount(novo_widget)

app = Nexus()
app.run()

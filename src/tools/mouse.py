"""Mouse-control tools exposed to the language agent."""

import usehid
from langchain.tools import tool

mouse = usehid.Mouse("Nexus")


@tool("move", description="Move o mouse para uma posição relativa à posição atual. Recebe um dicionário com as chaves 'x' e 'y', representando a quantidade de pixels a mover em cada direção.")
def move(x: int, y: int):
    """Move the pointer by the requested horizontal and vertical offsets."""
    mouse.move_by(x, y)

    return "OK"


@tool("click", description="Clica em um botão do mouse. Recebe uma string com o nome do botão a ser clicado.")
def click(button: str):
    """Click the requested mouse button."""
    mouse.click(button)

    return "OK"


@tool("scroll", description="Rola a roda do mouse. Recebe um inteiro representando a quantidade de rolagem (positivo para cima, negativo para baixo).")
def scroll(delta: int):
    """Scroll by the requested amount; positive values move upward."""
    mouse.scroll(delta)

    return "OK"


@tool("drag", description="Arrasta o mouse para uma posição. Recebe um dicionário com as chaves 'x' e 'y', representando a quantidade de pixels a mover em cada direção.")
def drag(destino: dict):
    """Hold the mouse button and move by the supplied x/y offsets."""
    mouse.press()

    mouse.move_by(destino["x"], destino["y"])

    return "OK"
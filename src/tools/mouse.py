import usehid
from langchain.tools import tool

mouse = usehid.Mouse("Nexus")

@tool("move", description="Move o mouse para uma posição relativa à posição atual. Recebe um dicionário com as chaves 'x' e 'y', representando a quantidade de pixels a mover em cada direção.")
def move(x: int, y: int):

    mouse.move_by(x, y)

    return "OK"


@tool("click", description="Clica em um botão do mouse. Recebe uma string com o nome do botão a ser clicado.")
def click(button: str):

    mouse.click(button)

    return "OK"


@tool("scroll", description="Rola a roda do mouse. Recebe um inteiro representando a quantidade de rolagem (positivo para cima, negativo para baixo).")
def scroll(delta: int):

    mouse.scroll(delta)

    return "OK"


@tool("drag", description="Arrasta o mouse para uma posição. Recebe um dicionário com as chaves 'x' e 'y', representando a quantidade de pixels a mover em cada direção.")
def drag(destino: dict):
    
    mouse.press()

    mouse.move_by(destino["x"], destino["y"])

    return "OK"
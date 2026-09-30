from datetime import datetime

from langchain.tools import tool


@tool(name_or_callable="Date", description="Pegar a data atual")
def data():
    dias = [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ]

    agora = datetime.now()  # noqa: DTZ005

    return agora.strftime("%d/%m/%Y"), dias[agora.weekday()]
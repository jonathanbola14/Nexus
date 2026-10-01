from datetime import datetime

from langchain.tools import tool


@tool(name_or_callable="Date", description="Pegar a data atual")
def data() -> tuple[str, str]:
    """Return the current date and its weekday name in Brazilian Portuguese."""
    weekdays = [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ]

    current_time = datetime.now()  # noqa: DTZ005

    return current_time.strftime("%d/%m/%Y"), weekdays[current_time.weekday()]
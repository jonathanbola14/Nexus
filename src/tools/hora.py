from datetime import datetime

from langchain.tools import tool


@tool(name_or_callable="Hour", description="Pegar a hora atual")
def hora() -> str:
    """Return the current local time in 24-hour format."""
    return datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
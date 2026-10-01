from importlib import import_module

from .data import data
from .hora import hora
from .print import print
from .web_search import web_search

__all__ = ["click", "data", "drag", "hora", "move", "print", "scroll", "web_search"]


def __getattr__(name):
    """Load mouse tools lazily to avoid initializing input devices on import."""
    if name in {"click", "drag", "move", "scroll"}:
        mouse_tools = import_module(".mouse", __name__)
        return getattr(mouse_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from importlib import import_module

from tools.data import data
from tools.hora import hora
from tools.print import print
from tools.web_search import web_search

__all__ = ["click", "data", "drag", "hora", "move", "print", "scroll", "web_search"]


def __getattr__(name):
	if name in {"click", "drag", "move", "scroll"}:
		mouse_tools = import_module(".mouse", __name__)
		return getattr(mouse_tools, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
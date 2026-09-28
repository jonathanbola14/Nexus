import importlib.util
from datetime import datetime as PythonDateTime
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def load_tool_module(name, filename):
    module_path = ROOT / "tools" / filename
    module_spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("day", "weekday"),
    [
        (1, "terça-feira"),
        (2, "quarta-feira"),
        (3, "quinta-feira"),
        (4, "sexta-feira"),
        (5, "sábado"),
        (6, "domingo"),
        (7, "segunda-feira"),
    ],
)
def test_data_returns_formatted_date_and_portuguese_weekday(monkeypatch, day, weekday):
    module = load_tool_module(f"data_tool_{day}", "data.py")
    frozen_datetime = PythonDateTime(2026, 9, day, 8, 9, 10)

    class FrozenDateTime:
        @staticmethod
        def now():
            return frozen_datetime

    monkeypatch.setattr(module, "datetime", FrozenDateTime)

    assert module.data.func() == (f"{day:02d}/09/2026", weekday)


def test_hora_returns_zero_padded_time(monkeypatch):
    module = load_tool_module("hour_tool", "hora.py")
    frozen_datetime = PythonDateTime(2026, 9, 28, 4, 5, 6)

    class FrozenDateTime:
        @staticmethod
        def now():
            return frozen_datetime

    monkeypatch.setattr(module, "datetime", FrozenDateTime)

    assert module.hora.func() == "04:05:06"
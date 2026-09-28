import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def load_main_module(monkeypatch):
    pyaudio_module = types.ModuleType("pyaudio")
    pyaudio_module.paInt16 = 8

    class FakeAudio:
        def open(self, **kwargs):
            return object()

    pyaudio_module.PyAudio = FakeAudio
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio_module)

    torch_module = types.ModuleType("torch")
    torch_module.set_num_threads = Mock()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    kokoro_module = types.ModuleType("kokoro")
    kokoro_module.KPipeline = Mock()
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_module)

    langchain_core_module = types.ModuleType("langchain_core")
    langchain_core_module.__path__ = []
    runnables_module = types.ModuleType("langchain_core.runnables")
    runnables_module.RunnableConfig = dict
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core_module)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", runnables_module)

    llm_module = types.ModuleType("LLM")
    llm_module.Agent = Mock()
    stt_module = types.ModuleType("STT")
    stt_module.Speech_to_Text = Mock()
    stt_module.load_stt_model = Mock()
    wakeword_module = types.ModuleType("WakeWord")
    wakeword_module.WakeWord = Mock()
    wakeword_module.configWakeWord = Mock()
    monkeypatch.setitem(sys.modules, "LLM", llm_module)
    monkeypatch.setitem(sys.modules, "STT", stt_module)
    monkeypatch.setitem(sys.modules, "WakeWord", wakeword_module)

    play_file_module = types.ModuleType("utils.play_file")
    play_file_module.Play = Mock()
    recorder_module = types.ModuleType("utils.recorder")
    recorder_module.collect_speech_frames = Mock()
    monkeypatch.setitem(sys.modules, "utils.play_file", play_file_module)
    monkeypatch.setitem(sys.modules, "utils.recorder", recorder_module)

    module_path = Path(__file__).parents[1] / "main.py"
    module_spec = importlib.util.spec_from_file_location("main_text_pipeline", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_split_ready_sentences_returns_complete_sentences_and_remainder(monkeypatch):
    main_module = load_main_module(monkeypatch)

    sentences, remainder = main_module.split_ready_sentences(
        "Olá! Tudo bem? Ainda estou falando"
    )

    assert sentences == ["Olá!", "Tudo bem?"]
    assert remainder == "Ainda estou falando"


def test_split_ready_sentences_keeps_abbreviations_and_decimal_numbers(monkeypatch):
    main_module = load_main_module(monkeypatch)

    sentences, remainder = main_module.split_ready_sentences(
        "O Dr. Silva pagou 3.14 reais. Próxima frase"
    )

    assert sentences == ["O Dr. Silva pagou 3.14 reais."]
    assert remainder == "Próxima frase"


def test_split_ready_sentences_waits_for_possible_decimal_continuation(monkeypatch):
    main_module = load_main_module(monkeypatch)

    sentences, remainder = main_module.split_ready_sentences("O valor é 3.")

    assert sentences == []
    assert remainder == "O valor é 3."
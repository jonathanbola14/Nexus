import importlib.util
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock


def load_main_module(monkeypatch):
    pyaudio_module = types.ModuleType("pyaudio")
    pyaudio_module.paInt16 = 8

    class FakeAudio:
        def __init__(self):
            self.opened = []

        def get_default_input_device_info(self):
            return {"defaultSampleRate": 44100}

        def open(self, **kwargs):
            self.opened.append(kwargs)
            return object()

    pyaudio_module.PyAudio = FakeAudio
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio_module)

    torch_module = types.ModuleType("torch")
    torch_module.set_num_threads = Mock()
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.Tensor = object
    torch_module.float32 = "float32"
    torch_module.__version__ = "0.0"
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
    src_module = types.ModuleType("src")
    src_module.__path__ = []
    monkeypatch.setitem(sys.modules, "src", src_module)
    monkeypatch.setitem(sys.modules, "src.LLM", llm_module)
    monkeypatch.setitem(sys.modules, "src.STT", stt_module)
    monkeypatch.setitem(sys.modules, "src.WakeWord", wakeword_module)

    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    play_file_module = types.ModuleType("utils.play_file")
    play_file_module.Play = Mock()
    recorder_module = types.ModuleType("utils.recorder")
    recorder_module.collect_speech_frames = Mock()
    monkeypatch.setitem(sys.modules, "utils", utils_module)
    audio_module = types.ModuleType("utils.audio")
    audio_module.resample_pcm16 = Mock()
    monkeypatch.setitem(sys.modules, "utils.audio", audio_module)
    increase_gain_module = types.ModuleType("utils.increase_gain")
    increase_gain_module.gain = Mock()
    monkeypatch.setitem(sys.modules, "utils.increase_gain", increase_gain_module)
    monkeypatch.setitem(sys.modules, "utils.play_file", play_file_module)
    monkeypatch.setitem(sys.modules, "utils.recorder", recorder_module)

    module_path = Path(__file__).parents[1] / "main.py"
    module_spec = importlib.util.spec_from_file_location("main_text_pipeline", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_split_ready_sentences_returns_complete_sentences_and_remainder(monkeypatch):
    main_module = load_main_module(monkeypatch)

    assert main_module.INPUT_RATE == 44100
    assert main_module.RATE == 16000
    assert main_module.mic.opened[0]["rate"] == 44100

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


def test_load_models_records_initialization_errors(monkeypatch):
    main_module = load_main_module(monkeypatch)

    def fail_loading(_load_state, _use_wake_word):
        raise RuntimeError("chave inválida")

    monkeypatch.setattr(main_module, "_load_models", fail_loading)
    load_state = {
        "lock": threading.Lock(),
        "message": "Iniciando",
        "error": None,
    }

    main_module.load_models(load_state, use_wake_word=False)

    assert isinstance(load_state["error"], RuntimeError)
    assert load_state["message"] == "Falha ao carregar modelos"
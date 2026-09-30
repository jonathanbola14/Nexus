import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def stub_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)

    if any(name == prefix or name.startswith(f"{prefix}.") for prefix in (
        "deepagents",
        "langchain",
        "langchain_core",
        "langgraph",
    )):
        module.__path__ = []

    package_name = name.rsplit(".", 1)[0] if "." in name else None
    if package_name:
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
        elif not hasattr(package, "__path__"):
            package.__path__ = []

    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_llm_module(monkeypatch):
    dotenv_values = Mock(return_value={"NVIDIA_API_KEY": "test-key"})
    stub_module(monkeypatch, "dotenv", dotenv_values=dotenv_values)

    agent_instance = object()
    create_deep_agent = Mock(return_value=agent_instance)
    stub_module(monkeypatch, "deepagents", create_deep_agent=create_deep_agent)
    filesystem_backend = Mock()
    stub_module(
        monkeypatch,
        "deepagents.backends.filesystem",
        FilesystemBackend=filesystem_backend,
    )
    memory_middleware = Mock()
    stub_module(
        monkeypatch,
        "deepagents.middleware.memory",
        MemoryMiddleware=memory_middleware,
    )

    shell_middleware = Mock()
    todo_middleware = Mock()
    stub_module(
        monkeypatch,
        "langchain.agents.middleware",
        ShellToolMiddleware=shell_middleware,
        TodoListMiddleware=todo_middleware,
    )

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    stub_module(monkeypatch, "langchain.messages", SystemMessage=SystemMessage)
    stub_module(monkeypatch, "langchain_core.runnables", Runnable=object)

    chat_model = object()
    chat_nvidia = Mock(return_value=chat_model)
    stub_module(
        monkeypatch,
        "langchain_nvidia_ai_endpoints",
        ChatNVIDIA=chat_nvidia,
    )

    checkpointer = object()
    in_memory_saver = Mock(return_value=checkpointer)
    stub_module(
        monkeypatch,
        "langgraph.checkpoint.memory",
        InMemorySaver=in_memory_saver,
    )
    store_instance = object()
    in_memory_store = Mock(return_value=store_instance)
    stub_module(
        monkeypatch,
        "langgraph.store.memory",
        InMemoryStore=in_memory_store,
    )

    tools_module = stub_module(monkeypatch, "src.tools")
    tools_module.data = object()
    tools_module.hora = object()
    tools_module.print = object()
    tools_module.web_search = object()

    module_path = Path(__file__).parents[1] / "src" / "LLM.py"
    module_spec = importlib.util.spec_from_file_location("llm_agent_under_test", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module, {
        "agent_instance": agent_instance,
        "chat_model": chat_model,
        "chat_nvidia": chat_nvidia,
        "checkpointer": checkpointer,
        "create_deep_agent": create_deep_agent,
        "dotenv_values": dotenv_values,
        "in_memory_saver": in_memory_saver,
        "store_instance": store_instance,
    }


def test_agent_builds_model_and_graph_with_expected_tools(monkeypatch):
    module, dependencies = load_llm_module(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    result = module.Agent()

    assert result is dependencies["agent_instance"]
    dependencies["chat_nvidia"].assert_called_once_with(
        model="z-ai/glm-5.3-flash",
        api_key="test-key",
        temperature=1,
        max_tokens=16384,
        seed=42,
    )
    dependencies["in_memory_saver"].assert_called_once_with()
    dependencies["create_deep_agent"].assert_called_once()

    agent_arguments = dependencies["create_deep_agent"].call_args.kwargs
    assert agent_arguments["model"] is dependencies["chat_model"]
    assert agent_arguments["tools"] == [
        module.hora,
        module.data,
        module.web_search,
        module.print,
    ]
    assert agent_arguments["checkpointer"] is dependencies["checkpointer"]
    assert agent_arguments["store"] is module.store
    assert agent_arguments["system_prompt"].content.startswith("Você é o Nexus")


def test_nvidia_api_key_uses_environment_before_dotenv(monkeypatch):
    module, dependencies = load_llm_module(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-key")

    assert module._get_nvidia_api_key() == "environment-key"
    dependencies["dotenv_values"].assert_not_called()


def test_nvidia_api_key_reports_missing_configuration(monkeypatch):
    module, dependencies = load_llm_module(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    dependencies["dotenv_values"].return_value = {}

    try:
        module._get_nvidia_api_key()
    except RuntimeError as error:
        assert "NVIDIA_API_KEY ausente" in str(error)
    else:
        raise AssertionError("Esperava erro quando a chave NVIDIA não está configurada")
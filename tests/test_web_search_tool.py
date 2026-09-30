import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load_web_search_module(monkeypatch):
    firecrawl_package = types.ModuleType("firecrawl")
    firecrawl_module = types.ModuleType("firecrawl.v2")
    firecrawl_module.FirecrawlClient = object
    firecrawl_package.v2 = firecrawl_module
    monkeypatch.setitem(sys.modules, "firecrawl", firecrawl_package)
    monkeypatch.setitem(sys.modules, "firecrawl.v2", firecrawl_module)

    module_path = Path(__file__).parents[1] / "src" / "tools" / "web_search.py"
    module_spec = importlib.util.spec_from_file_location("web_search_tool", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_web_search_returns_web_results_and_forwards_arguments(monkeypatch):
    module = load_web_search_module(monkeypatch)
    web_results = [{"title": "Resultado"}]
    client = Mock()
    client.search.return_value = SimpleNamespace(web=web_results)
    monkeypatch.setattr(module, "FirecrawlClient", lambda: client)

    result = module.web_search.func("notícias locais", limit=3)

    assert result is web_results
    client.search.assert_called_once_with("notícias locais", limit=6)
import base64
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

module_path = Path(__file__).parents[1] / "src" / "tools" / "print.py"
module_spec = importlib.util.spec_from_file_location("print_tool", module_path)
print_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(print_module)


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGB", "red"),
        ("RGBA", (0, 255, 0, 128)),
        ("L", 128),
    ],
)
def test_print_returns_prompt_and_base64_jpeg(monkeypatch, mode, color):
    screenshot = Image.new(mode, (2, 2), color=color)
    captures = []

    def capture_screenshot():
        captures.append(None)
        return screenshot

    monkeypatch.setattr(print_module.pyg, "screenshot", capture_screenshot)

    result = print_module.print.invoke({})

    assert result[0] == {
        "type": "text",
        "text": "Describe the content of this image.",
    }
    assert result[1]["type"] == "image"
    assert result[1]["mime_type"] == "image/jpeg"

    image_bytes = base64.b64decode(result[1]["base64"], validate=True)
    assert image_bytes.startswith(b"\xff\xd8")

    decoded_image = Image.open(BytesIO(image_bytes))
    assert decoded_image.format == "JPEG"
    assert decoded_image.mode == "RGB"
    assert decoded_image.size == (2, 2)
    assert len(captures) == 1
import base64
from io import BytesIO

import pyautogui as pyg
from langchain.tools import tool


@tool()
def print():
    """Capture a tela e retorne a imagem codificada em JPEG e Base64."""
    imagen = pyg.screenshot().convert("RGB")
    buffer = BytesIO()
    imagen.save(buffer, format="JPEG", quality=85)

    imagen_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": imagen_base64,
            "mime_type": "image/jpeg",
        },
    ]
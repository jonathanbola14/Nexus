import base64
from io import BytesIO

import pyautogui as pyg
from langchain.tools import tool


@tool()
def print():
    """Capture the screen and return the image encoded as JPEG and Base64."""
    image = pyg.screenshot().convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": "image/jpeg",
        },
    ]
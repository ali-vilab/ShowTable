import base64
from PIL import Image
from io import BytesIO

def base64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_image(img: Image.Image, path: str) -> None:
    img.save(path, format="PNG")

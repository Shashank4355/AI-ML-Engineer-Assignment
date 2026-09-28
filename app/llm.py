"""Thin client for the self-hosted vLLM server (OpenAI-compatible API)."""
import base64
import io

from PIL import Image
from openai import OpenAI
from app.config import VLLM_BASE_URL, VLLM_MODEL

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


def chat(prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _shrink(image_bytes: bytes, max_side: int = 768) -> bytes:
    """Downscale large images so their vision tokens fit in the context window."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def chat_image(image_bytes: bytes, prompt: str, mime: str = "image/png", max_tokens: int = 512) -> str:
    b64 = base64.b64encode(_shrink(image_bytes)).decode()
    mime = "image/png"
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

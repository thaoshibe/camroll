"""OpenAI VLM + LLM clients.

Requires `pip install -r requirements.txt` and `OPENAI_API_KEY` env var.
Custom OPENAI_BASE_URL is honored (useful for OpenAI-compatible proxies / vLLM).
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from camroll_agent.llm.base import LLMClient, VLMClient

DEFAULT_VLM_MODEL = "gpt-4o"
DEFAULT_LLM_MODEL = "gpt-4o"
_MAX_IMAGE_BYTES = 19 * 1024 * 1024


def _load_image_b64(path: str) -> tuple[str, str]:
    from PIL import Image

    raw = Path(path).read_bytes()
    if len(raw) <= _MAX_IMAGE_BYTES:
        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return base64.b64encode(raw).decode("utf-8"), mime

    # Shrink big files (JPEG re-encode, then downscale if still too big).
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    quality = 85
    data = raw
    while len(data) > _MAX_IMAGE_BYTES and quality >= 30:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        quality -= 10
    while len(data) > _MAX_IMAGE_BYTES:
        w, h = img.size
        img = img.resize((max(w // 2, 1), max(h // 2, 1)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        data = buf.getvalue()
    return base64.b64encode(data).decode("utf-8"), "image/jpeg"


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI backend requires the openai SDK. Run: pip install -r requirements.txt"
        ) from exc
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


class OpenAIVLM(VLMClient):
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_VLM_MODEL
        self._client = _client()

    def generate(self, prompt: str, image_paths: list[str]) -> str:
        if not image_paths:
            return self._chat([{"type": "text", "text": prompt}])

        # Send every image inline so the model actually sees all of them.
        content: list[dict] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            b64, mime = _load_image_b64(path)
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        return self._chat(content)

    def _chat(self, content: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""


class OpenAILLM(LLMClient):
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_LLM_MODEL
        self._client = _client()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        tool_choice: str | dict = "auto",
    ) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        out: dict[str, Any] = {"role": "assistant", "content": msg.content}
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in tcs
            ]
        return out

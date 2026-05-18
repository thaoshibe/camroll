"""Google Gemini VLM + LLM clients.

Requires `pip install camroll-agent[gemini]` and `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`) env var.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from camroll_agent.llm.base import LLMClient, VLMClient

DEFAULT_VLM_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_MODEL = "gemini-2.5-pro"
_MAX_IMAGE_BYTES = 19 * 1024 * 1024


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return key


def _client():
    try:
        from google import genai
        from google.genai import types as gt
    except ImportError as exc:
        raise ImportError(
            "Gemini backend requires `pip install camroll-agent[gemini]`."
        ) from exc
    return genai.Client(
        api_key=_api_key(),
        http_options=gt.HttpOptions(timeout=600_000),
    )


def _load_image(path: str):
    """Read an image as a PIL Image, shrinking it if it's over the byte limit."""
    from PIL import Image

    if os.path.getsize(path) <= _MAX_IMAGE_BYTES:
        return Image.open(path)
    img = Image.open(path).convert("RGB")
    quality = 85
    while quality >= 30:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if len(buf.getvalue()) <= _MAX_IMAGE_BYTES:
            buf.seek(0)
            return Image.open(buf).copy()
        quality -= 10
    while True:
        w, h = img.size
        img = img.resize((max(w // 2, 1), max(h // 2, 1)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        if len(buf.getvalue()) <= _MAX_IMAGE_BYTES:
            buf.seek(0)
            return Image.open(buf).copy()


class GeminiVLM(VLMClient):
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_VLM_MODEL
        self._client = _client()

    def generate(self, prompt: str, image_paths: list[str]) -> str:
        contents: list[Any] = [prompt]
        if not image_paths:
            resp = self._client.models.generate_content(
                model=self.model, contents=contents,
            )
            return resp.text or ""

        if len(image_paths) == 1:
            contents.extend([
                "Image 1: current album image to process. "
                "Base the caption and event decision on this image.",
                _load_image(str(Path(image_paths[0]))),
            ])
        else:
            for idx, path in enumerate(image_paths, start=1):
                if idx == len(image_paths):
                    label = (
                        f"Image {idx}: current album image to process. "
                        "Use this image for the caption and event decision."
                    )
                else:
                    label = (
                        f"Image {idx}: reference/profile image only. "
                        "Do not caption this image and do not base the event on it."
                    )
                contents.extend([label, _load_image(str(Path(path)))])
        resp = self._client.models.generate_content(
            model=self.model, contents=contents,
        )
        return resp.text or ""


class GeminiLLM(LLMClient):
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
        from google.genai import types as gt

        # Pull the system message out and convert the rest into Gemini's format.
        system_text = ""
        contents: list = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_text = (system_text + "\n" + m["content"]).strip()
                continue
            if role == "tool":
                contents.append(gt.Content(
                    role="user",
                    parts=[gt.Part(function_response=gt.FunctionResponse(
                        name=m.get("name", "tool"),
                        response={"result": m.get("content", "")},
                    ))],
                ))
                continue
            if role == "assistant" and m.get("tool_calls"):
                parts = []
                if m.get("content"):
                    parts.append(gt.Part(text=m["content"]))
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args or "{}")
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(gt.Part(function_call=gt.FunctionCall(
                        name=fn["name"], args=args,
                    )))
                contents.append(gt.Content(role="model", parts=parts))
                continue
            parts = [gt.Part(text=m.get("content") or "")]
            contents.append(gt.Content(
                role="user" if role != "assistant" else "model",
                parts=parts,
            ))

        cfg_kwargs: dict[str, Any] = {}
        if system_text:
            cfg_kwargs["system_instruction"] = system_text
        if tools:
            cfg_kwargs["tools"] = _to_gemini_tools(tools)
            cfg_kwargs["tool_config"] = gt.ToolConfig(
                function_calling_config=gt.FunctionCallingConfig(mode="AUTO"),
            )

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gt.GenerateContentConfig(**cfg_kwargs),
        )
        cand = resp.candidates[0] if resp.candidates else None
        content = cand.content if cand else None

        if content is None:
            return {"role": "assistant", "content": "", "tool_calls": []}

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for part in (content.parts or []):
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append({
                    "id": f"call_{fc.name}_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(dict(fc.args or {})),
                    },
                })
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        out: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) or None,
        }
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out


def _to_gemini_tools(openai_tools: list[dict]):
    """Convert OpenAI-style tool schemas to Gemini Tool / FunctionDeclaration."""
    from google.genai import types as gt

    type_map = {
        "string":  gt.Type.STRING,
        "number":  gt.Type.NUMBER,
        "integer": gt.Type.INTEGER,
        "boolean": gt.Type.BOOLEAN,
        "array":   gt.Type.ARRAY,
        "object":  gt.Type.OBJECT,
    }

    def to_schema(prop_schema: dict) -> "gt.Schema":
        ptype = type_map.get(str(prop_schema.get("type", "string")).lower(),
                             gt.Type.STRING)
        kwargs: dict = {"type": ptype,
                        "description": prop_schema.get("description", "")}
        if ptype == gt.Type.ARRAY:
            items = prop_schema.get("items") or {"type": "string"}
            kwargs["items"] = to_schema(items)
        if "enum" in prop_schema:
            kwargs["enum"] = list(prop_schema["enum"])
        return gt.Schema(**kwargs)

    declarations = []
    for tool in openai_tools:
        fn = tool["function"]
        params = fn.get("parameters", {})
        properties = {
            name: to_schema(s) for name, s in params.get("properties", {}).items()
        }
        schema = gt.Schema(
            type=gt.Type.OBJECT,
            properties=properties,
            required=params.get("required", []),
        )
        declarations.append(gt.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=schema,
        ))
    return [gt.Tool(function_declarations=declarations)]

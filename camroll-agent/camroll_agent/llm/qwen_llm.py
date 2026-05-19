"""Local Qwen LLM client for the ReAct agent (tool-calling).

Requires `pip install -r requirements_local.txt` and a CUDA GPU.

Supported models:
  Qwen/Qwen2.5-1.5B-Instruct          CPU-friendly, ~70% tool-use reliability
  Qwen/Qwen2.5-7B-Instruct            recommended on GPU, ~90% reliability
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  large MoE, needs ~15 GB VRAM

Override the default via the CAMROLL_LLM env var.

Two tool-call formats are supported and auto-detected:
  Qwen2.5-Instruct  — Hermes-style JSON inside <tool_call>...</tool_call>
  Qwen3-Coder       — XML-style <function=name><parameter=key>value</parameter>
Both are normalised to OpenAI-shaped tool_calls dicts.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from camroll_agent.llm.base import LLMClient

_CUDA_AVAILABLE = False
try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except Exception:
    pass

DEFAULT_MODEL = os.environ.get(
    "CAMROLL_LLM",
    # GPU: Qwen2.5-Coder-7B — good tool-use reliability, ~15 GB VRAM
    # CPU: tiny fallback
    "Qwen/Qwen2.5-Coder-7B-Instruct" if _CUDA_AVAILABLE
    else "Qwen/Qwen2.5-1.5B-Instruct",
)

# ── Qwen2.5 Hermes-JSON patterns ──────────────────────────────────────────────
_TOOL_RE_FULL   = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_RE_OPEN   = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)
_TOOL_RE_FENCE  = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"^\s*(\{.*?\})\s*$", re.DOTALL)

# ── Qwen3-Coder XML patterns ───────────────────────────────────────────────────
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL,
)
_XML_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][\w\-]*)\s*>\s*(.*?)\s*(?:</function>|$)",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][\w\-]*)\s*>\s*(.*?)\s*"
    r"(?:</parameter>|(?=<parameter|</function|</tool_call|$))",
    re.DOTALL,
)


class QwenLLM(LLMClient):
    """Qwen2.5-Instruct / Qwen3-Coder as a drop-in LLMClient."""

    def __init__(self, model_name: str | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local LLM requires torch + transformers. "
                "Run: pip install -r requirements_local.txt"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "QwenLLM requires a CUDA GPU. Use --llm-backend openai or gemini instead."
            )

        self.model_name = model_name or DEFAULT_MODEL
        is_fp8 = "FP8" in self.model_name or "fp8" in self.model_name
        dtype = "auto" if is_fp8 else torch.bfloat16

        print(f"[QwenLLM] loading {self.model_name}…", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = dict(torch_dtype=dtype, trust_remote_code=True)
        if is_fp8 or "30B" in self.model_name or "32B" in self.model_name:
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs,
        )
        if "device_map" not in load_kwargs:
            self.model = self.model.to("cuda")
        self.model.eval()
        print("[QwenLLM] ready.", flush=True)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        tool_choice: str | dict = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> dict:
        import torch

        prepped = _normalize_tool_call_arguments(messages)
        with torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                prepped,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            prompt_len = inputs.input_ids.shape[1]
            do_sample = temperature > 0
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True,
            ).strip()

        short = raw if len(raw) <= 1200 else raw[:1200] + "…[truncated]"
        print(f"[QwenLLM] raw:\n{short}\n[/QwenLLM]", flush=True)
        return _parse(raw, tools=tools)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse(raw: str, tools: list[dict] | None = None) -> dict:
    msg = _parse_xml_tool_calls(raw, tools=tools or [])
    if msg.get("tool_calls"):
        return msg
    return _parse_hermes_json_tool_calls(raw)


def _parse_xml_tool_calls(raw: str, tools: list[dict]) -> dict:
    schema_lookup: dict[str, dict[str, dict]] = {}
    for t in tools:
        fn = (t.get("function") or {})
        name = fn.get("name")
        if name:
            schema_lookup[name] = ((fn.get("parameters") or {}).get("properties") or {})

    tool_calls: list[dict] = []
    consumed_spans: list[tuple[int, int]] = []

    for m_call in _XML_TOOL_CALL_RE.finditer(raw):
        inner = m_call.group(1)
        for m_fn in _XML_FUNCTION_RE.finditer(inner):
            fn_name = m_fn.group(1).strip()
            args: dict[str, Any] = {}
            for m_p in _XML_PARAM_RE.finditer(m_fn.group(2)):
                pname = m_p.group(1).strip()
                args[pname] = _coerce_param(m_p.group(2).strip(), schema_lookup.get(fn_name, {}).get(pname))
            tool_calls.append({
                "id": f"call_{fn_name}_{len(tool_calls)}",
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
        consumed_spans.append(m_call.span())

    content = _remove_spans(raw, consumed_spans)
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _parse_hermes_json_tool_calls(raw: str) -> dict:
    tool_calls: list[dict] = []
    consumed_spans: list[tuple[int, int]] = []

    def _add(obj: dict, span: tuple[int, int]) -> bool:
        name = obj.get("name")
        if not name:
            return False
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append({
            "id": f"call_{name}_{len(tool_calls)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        consumed_spans.append(span)
        return True

    for m in _TOOL_RE_FULL.finditer(raw):
        try:
            _add(json.loads(m.group(1)), m.span())
        except json.JSONDecodeError:
            pass

    if not tool_calls:
        for m in _TOOL_RE_OPEN.finditer(raw):
            obj = _try_truncated_json(m.group(1))
            if obj:
                _add(obj, m.span())

    if not tool_calls:
        for m in _TOOL_RE_FENCE.finditer(raw):
            try:
                obj = json.loads(m.group(1))
                if "name" in obj:
                    _add(obj, m.span())
            except json.JSONDecodeError:
                pass

    if not tool_calls:
        m = _BARE_OBJECT_RE.match(raw)
        if m:
            try:
                obj = json.loads(m.group(1))
                if obj and "name" in obj:
                    _add(obj, m.span())
            except json.JSONDecodeError:
                pass

    content = _remove_spans(raw, consumed_spans)
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str | None:
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + text[e:]
    return text.strip() or None


def _normalize_tool_call_arguments(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            out.append(msg)
            continue
        new_calls = []
        for tc in msg["tool_calls"]:
            fn = (tc.get("function") or {}).copy()
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    fn["arguments"] = {}
            new_calls.append({**tc, "function": fn})
        out.append({**msg, "tool_calls": new_calls})
    return out


def _try_truncated_json(s: str) -> dict | None:
    s = s.strip()
    for end in range(len(s), 0, -1):
        try:
            return json.loads(s[:end])
        except json.JSONDecodeError:
            continue
    return None


def _coerce_param(value: str, schema: dict | None) -> Any:
    if not schema:
        v = value.strip()
        if v.startswith(("[", "{")):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return value

    t = (schema.get("type") or "string").lower()
    v = value.strip()

    if t == "integer":
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    if t in ("number", "float"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    if t == "boolean":
        return v.lower() in ("true", "1", "yes", "y")
    if t == "array":
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except json.JSONDecodeError:
                pass
            try:
                import ast
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except (ValueError, SyntaxError):
                pass
            v = v[1:-1]
        if v.startswith(("'", '"')) and v.endswith(("'", '"')) and len(v) >= 2:
            v = v[1:-1]
        items = [p.strip().strip("'\"") for p in re.split(r"[,\n]", v)]
        return [x for x in items if x]
    if t == "object":
        if v.startswith("{"):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
    return v

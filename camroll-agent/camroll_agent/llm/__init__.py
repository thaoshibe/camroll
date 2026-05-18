"""Pluggable VLM and LLM backends.

Two small interfaces:

  VLMClient.generate(prompt, image_paths) -> str
      Used by Stage 1 (captioning) and by the agent's view_image tool.

  LLMClient.chat(messages, tools) -> assistant_message
      Used by the ReAct agent. Returns an OpenAI-style message dict:
        {"role": "assistant",
         "content": str | None,
         "tool_calls": [{"id", "type": "function",
                         "function": {"name", "arguments"}}, ...] | None}

Built-in backends:
  - openai   pip install -r requirements.txt        env: OPENAI_API_KEY
  - gemini   pip install -r requirements.txt        env: GEMINI_API_KEY
  - local    pip install -r requirements_local.txt  (HuggingFace, needs GPU)

You can plug in any other backend by subclassing VLMClient or LLMClient.
"""
from __future__ import annotations

from camroll_agent.llm.base import LLMClient, VLMClient

__all__ = ["LLMClient", "VLMClient", "build_vlm", "build_llm"]


def build_vlm(backend: str, model: str | None = None, **kwargs) -> VLMClient:
    """Factory for VLM clients. backend ∈ {openai, gemini, local}."""
    name = backend.lower().strip()
    if name in ("openai", "gpt"):
        from camroll_agent.llm.openai_client import OpenAIVLM
        return OpenAIVLM(model=model, **kwargs)
    if name == "gemini":
        from camroll_agent.llm.gemini_client import GeminiVLM
        return GeminiVLM(model=model, **kwargs)
    if name in ("local", "hf", "huggingface"):
        from camroll_agent.llm.local_client import LocalVLM
        return LocalVLM(model_id=model, **kwargs)
    raise ValueError(f"unknown VLM backend: {backend!r}")


def build_llm(backend: str, model: str | None = None, **kwargs) -> LLMClient:
    """Factory for LLM clients (tool-calling). backend ∈ {openai, gemini}."""
    name = backend.lower().strip()
    if name in ("openai", "gpt"):
        from camroll_agent.llm.openai_client import OpenAILLM
        return OpenAILLM(model=model, **kwargs)
    if name == "gemini":
        from camroll_agent.llm.gemini_client import GeminiLLM
        return GeminiLLM(model=model, **kwargs)
    raise ValueError(f"unknown LLM backend: {backend!r}")

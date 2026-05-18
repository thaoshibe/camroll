"""Abstract base classes for VLM and LLM clients."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VLMClient(ABC):
    """A vision-language model that can read images and produce text.

    Convention for image_paths:
      - All paths before the last are reference / profile images (identity context).
      - The last path is the "current" image the prompt is about.
    Single-image calls (e.g. view_image with one id) are fine.
    """

    @abstractmethod
    def generate(self, prompt: str, image_paths: list[str]) -> str:
        ...


class LLMClient(ABC):
    """A text-only LLM with OpenAI-style tool-calling.

    The agent expects this interface (matches OpenAI's chat-completions
    message format so the rest of the code is provider-agnostic).
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        tool_choice: str | dict = "auto",
    ) -> dict:
        """Returns an OpenAI-shaped assistant message:
          {"role": "assistant",
           "content": str | None,
           "tool_calls": [{"id", "type": "function",
                           "function": {"name", "arguments"}}, ...]?}
        """
        ...

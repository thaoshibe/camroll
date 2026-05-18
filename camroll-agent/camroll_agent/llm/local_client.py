"""Local HuggingFace VLM client (Qwen-VL, Kimi-VL, generic image-text-to-text).

Requires `pip install camroll-agent[local]` and a CUDA GPU.

Supported families (auto-detected):
  - qwen-vl  Qwen2.5-VL, Qwen3-VL  (uses qwen_vl_utils when available)
  - kimi     moonshotai Kimi-VL
  - generic  any HF image-text-to-text model
"""
from __future__ import annotations

import re
from typing import Any

from camroll_agent.llm.base import VLMClient

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1024


def _detect_family(model_id: str) -> str:
    lower = model_id.lower()
    if "qwen" in lower and "vl" in lower:
        return "qwen-vl"
    if "kimi" in lower or "moonshot" in lower:
        return "kimi"
    return "generic"


class LocalVLM(VLMClient):
    def __init__(
        self,
        model_id: str | None = None,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        attn_implementation: str | None = "sdpa",
    ):
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Local VLM requires `pip install camroll-agent[local]`."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LocalVLM requires a CUDA GPU. Use a cloud backend instead "
                "(openai / gemini) if no GPU is available."
            )

        self.model_id = model_id or DEFAULT_MODEL_ID
        self.max_new_tokens = max_new_tokens
        self._family = _detect_family(self.model_id)

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        load_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        print(f"[local-vlm] loading {self.model_id} (family={self._family})…",
              flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self.model = self._load_model(load_kwargs)
        self.model.eval()
        print("[local-vlm] ready.", flush=True)

        self._has_qwen_vl_utils = False
        if self._family == "qwen-vl":
            try:
                import qwen_vl_utils  # noqa: F401
                self._has_qwen_vl_utils = True
            except ImportError:
                print("[local-vlm] qwen_vl_utils not found — PIL fallback.",
                      flush=True)

    def _load_model(self, load_kwargs: dict):
        import transformers
        candidate_names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
        last_exc: Exception = RuntimeError("no auto-class found")
        for name in candidate_names:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(self.model_id, **load_kwargs)
            except (ValueError, KeyError) as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"Could not load {self.model_id} with any known auto-class: {last_exc}"
        )

    def generate(self, prompt: str, image_paths: list[str]) -> str:
        messages = self._build_messages(prompt, image_paths)
        if self._family == "qwen-vl" and self._has_qwen_vl_utils:
            return _strip_thinking(self._gen_qwen(messages, image_paths))
        return _strip_thinking(self._gen_generic(messages, image_paths))

    def _build_messages(self, prompt: str, image_paths: list[str]) -> list[dict]:
        content: list[dict] = []
        for i, path in enumerate(image_paths):
            if i < len(image_paths) - 1:
                label = (
                    f"Image {i + 1} (profile / reference — identity context, "
                    "do NOT caption this image): "
                )
            else:
                label = (
                    f"Image {i + 1} (current album image — write the caption "
                    "and event decision based on THIS image): "
                )
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": "\n" + prompt})
        return [{"role": "user", "content": content}]

    def _gen_qwen(self, messages: list[dict], image_paths: list[str]) -> str:
        import torch
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        pil_cache = {p: PILImage.open(p).convert("RGB") for p in image_paths}
        messages_pil = [
            {
                **msg,
                "content": [
                    {**item, "image": pil_cache[item["image"]]}
                    if item.get("type") == "image" else item
                    for item in msg["content"]
                ],
            }
            for msg in messages
        ]

        kwargs: dict[str, Any] = {
            "tokenize": False, "add_generation_prompt": True,
        }
        try:
            text = self.processor.apply_chat_template(
                messages_pil, **kwargs, enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(messages_pil, **kwargs)

        image_inputs, video_inputs = process_vision_info(messages_pil)
        if not image_inputs:
            raise ValueError(f"no images recognized in: {image_paths}")

        proc_kwargs: dict[str, Any] = {
            "text": [text], "images": image_inputs,
            "padding": True, "return_tensors": "pt",
        }
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with torch.no_grad():
            ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        in_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(ids[:, in_len:],
                                           skip_special_tokens=True)[0]

    def _gen_generic(self, messages: list[dict], image_paths: list[str]) -> str:
        import torch
        from PIL import Image as PILImage

        stripped = [
            {
                **msg,
                "content": [
                    {"type": "image"} if item.get("type") == "image" else item
                    for item in msg["content"]
                ],
            }
            for msg in messages
        ]
        text = self.processor.apply_chat_template(
            stripped, tokenize=False, add_generation_prompt=True,
        )
        pil_images = [PILImage.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=text,
            images=pil_images or None,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        in_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(ids[:, in_len:],
                                           skip_special_tokens=True)[0]


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

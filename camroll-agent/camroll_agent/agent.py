"""ReAct-style agent loop.

The agent reasons over the 5 atomic tools and terminates by emitting plain
text (no `answer` tool). Adds:
  - Required `thought` field on every tool (enforced in schemas)
  - Consistent observation formatter (prompts.format_observation)
  - Per-turn budget line in the context ("[step K/N | view_image budget: X/Y]")
  - Separate caps: max_steps and max_view_image_calls

Usage:

    from camroll_agent import Agent
    agent = Agent(memory_dir="memory/")
    print(agent.ask("When did I go to Lake Michigan?"))
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator

from camroll_agent import tools as atom_tools
from camroll_agent.llm import LLMClient, VLMClient, build_llm
from camroll_agent.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FREEFORM_SUFFIX,
    SYSTEM_PROMPT_MCQ_SUFFIX,
    format_observation,
)
from camroll_agent.schemas import TOOLS

DEFAULT_MAX_STEPS = 25
DEFAULT_MAX_VIEW_IMAGE_CALLS = 5


# ── result type ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    final_text: str
    tool_trace: list[dict]
    steps: int
    view_image_calls: int
    stopped_reason: str
    latency_s: float

    def __str__(self) -> str:
        return self.final_text


# ── tool dispatch ────────────────────────────────────────────────────────────

def _execute_tool(
    fn_name: str,
    fn_args: dict,
    memory_dir: str,
    image_client: VLMClient | None,
) -> dict | str:
    thought = fn_args.get("thought", "")
    try:
        if fn_name == "search":
            return atom_tools.search(
                thought=thought,
                query=fn_args.get("query", ""),
                memory_dir=memory_dir,
                top_k=int(fn_args.get("top_k", 10)),
                kind=fn_args.get("kind", "both") or "both",
                date_from=fn_args.get("date_from") or None,
                date_to=fn_args.get("date_to") or None,
            )
        if fn_name == "grep":
            return atom_tools.grep(
                thought=thought,
                query=fn_args.get("query", ""),
                memory_dir=memory_dir,
                kind=fn_args.get("kind", "both") or "both",
                top_k=int(fn_args.get("top_k", 10)),
                date_from=fn_args.get("date_from") or None,
                date_to=fn_args.get("date_to") or None,
            )
        if fn_name == "list_by_date":
            return atom_tools.list_by_date(
                thought=thought,
                memory_dir=memory_dir,
                date_from=fn_args.get("date_from") or None,
                date_to=fn_args.get("date_to") or None,
                kind=fn_args.get("kind", "both") or "both",
                location=fn_args.get("location") or None,
                person=fn_args.get("person") or None,
                limit=int(fn_args.get("limit", 50)),
            )
        if fn_name == "get":
            return atom_tools.get(
                thought=thought,
                id=fn_args.get("id", ""),
                memory_dir=memory_dir,
            )
        if fn_name == "view_image":
            image_ids = fn_args.get("image_ids") or []
            if isinstance(image_ids, str):
                image_ids = [image_ids]
            return atom_tools.view_image(
                thought=thought,
                image_ids=image_ids,
                prompt=fn_args.get("prompt", ""),
                memory_dir=memory_dir,
                image_client=image_client,
            )
        return {"error": f"unknown tool: {fn_name}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"tool {fn_name} raised: {exc}"}


def _budget_line(step: int, max_steps: int, vi_used: int, vi_max: int) -> str:
    return (f"[step {step}/{max_steps}  |  "
            f"view_image budget: {vi_max - vi_used}/{vi_max} remaining]")


# ── public Agent class ───────────────────────────────────────────────────────

class Agent:
    """A camera-roll agent that answers questions about a photo library.

    Args:
        memory_dir: directory containing metadata.sqlite + vector_store/
                    (output of camroll_agent.index.run).
        llm: an LLMClient instance. If None, builds one from `llm_backend`.
        vlm: a VLMClient for the view_image tool. If None and view_image is
             called, the tool returns an error message but the agent keeps
             running. Pass build_vlm("openai") (etc.) to enable.
        llm_backend: which built-in LLM to use ("openai" or "gemini").
                     Ignored if `llm` is set.
        llm_model: model name for the chosen backend.
        max_steps: ReAct loop step cap.
        max_view_image_calls: per-question cap on the expensive view_image tool.
    """

    def __init__(
        self,
        memory_dir: str,
        *,
        llm: LLMClient | None = None,
        vlm: VLMClient | None = None,
        llm_backend: str = "openai",
        llm_model: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_view_image_calls: int = DEFAULT_MAX_VIEW_IMAGE_CALLS,
    ):
        self.memory_dir = memory_dir
        self.llm = llm or build_llm(llm_backend, llm_model)
        self.vlm = vlm
        self.max_steps = max_steps
        self.max_view_image_calls = max_view_image_calls

    def ask(
        self,
        question: str,
        *,
        choices: dict[str, str] | None = None,
        eval_mode: str = "freeform",
    ) -> AgentResult:
        """Ask a question; returns the agent's answer + tool trace.

        Args:
            question: the user question.
            choices: for multiple-choice eval, a {letter: text} mapping.
            eval_mode: "freeform" or "multiple_choice".
        """
        return _run(
            question=question,
            choices=choices,
            eval_mode=eval_mode,
            memory_dir=self.memory_dir,
            llm=self.llm,
            vlm=self.vlm,
            max_steps=self.max_steps,
            max_view_image_calls=self.max_view_image_calls,
        )

    def ask_streaming(
        self,
        question: str,
        *,
        choices: dict[str, str] | None = None,
        eval_mode: str = "freeform",
    ) -> Iterator[tuple[str, dict]]:
        """Streaming variant. Yields (event_type, data) tuples:
            ("status", {"message": str})
            ("thought", {"step": int, "text": str})
            ("tool_call", {"step": int, "tool": str, "args": dict})
            ("tool_result", {"step": int, "tool": str, "observation": str})
            ("answer", {"response": str, "steps": int})
        """
        return _run_events(
            question=question,
            choices=choices,
            eval_mode=eval_mode,
            memory_dir=self.memory_dir,
            llm=self.llm,
            vlm=self.vlm,
            max_steps=self.max_steps,
            max_view_image_calls=self.max_view_image_calls,
        )


# ── inner loop ───────────────────────────────────────────────────────────────

def _format_question(question: str, choices: dict[str, str] | None, eval_mode: str) -> str:
    if eval_mode == "freeform" or not choices:
        return f"Question: {question}"
    choices_text = "\n".join(f"  {L}. {t}" for L, t in choices.items())
    return f"Question: {question}\nChoices:\n{choices_text}"


def _run(
    *,
    question: str,
    choices: dict[str, str] | None,
    eval_mode: str,
    memory_dir: str,
    llm: LLMClient,
    vlm: VLMClient | None,
    max_steps: int,
    max_view_image_calls: int,
) -> AgentResult:
    is_freeform = eval_mode == "freeform"
    system_prompt = SYSTEM_PROMPT + (
        SYSTEM_PROMPT_FREEFORM_SUFFIX if is_freeform else SYSTEM_PROMPT_MCQ_SUFFIX
    )
    user_prompt = _format_question(question, choices, eval_mode)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    tool_trace: list[dict] = []
    vi_used = 0
    step = 0
    t0 = time.perf_counter()

    while step < max_steps:
        budget = _budget_line(step + 1, max_steps, vi_used, max_view_image_calls)
        messages.append({"role": "user", "content": budget})

        assistant_msg = llm.chat(messages, tools=TOOLS)
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            return AgentResult(
                final_text=(assistant_msg.get("content") or "").strip(),
                tool_trace=tool_trace,
                steps=step,
                view_image_calls=vi_used,
                stopped_reason="ok",
                latency_s=round(time.perf_counter() - t0, 3),
            )

        step += 1
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name == "view_image" and vi_used >= max_view_image_calls:
                obs = (
                    f"[view_image] You have already viewed {vi_used} photo set(s) "
                    f"and have enough visual context. Use the visual details from "
                    f"your earlier view_image observations to answer the question "
                    f"now. Do not mention this message in your final answer."
                )
                tool_trace.append({
                    "tool": fn_name, "args": fn_args,
                    "result": {"note": "view_image quota reached"},
                })
            else:
                result = _execute_tool(fn_name, fn_args, memory_dir, vlm)
                if fn_name == "view_image":
                    vi_used += 1
                tool_trace.append({"tool": fn_name, "args": fn_args, "result": result})
                obs = format_observation(fn_name, fn_args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": obs,
            })

    # Step cap reached — force a plain-text answer.
    messages.append({
        "role": "user",
        "content": (
            f"You have used all {max_steps} steps. Stop calling tools and "
            "write your final answer now as plain text."
        ),
    })
    final_msg = llm.chat(messages, tools=None)
    return AgentResult(
        final_text=(final_msg.get("content") or "").strip(),
        tool_trace=tool_trace,
        steps=step,
        view_image_calls=vi_used,
        stopped_reason="max_steps",
        latency_s=round(time.perf_counter() - t0, 3),
    )


def _run_events(
    *,
    question: str,
    choices: dict[str, str] | None,
    eval_mode: str,
    memory_dir: str,
    llm: LLMClient,
    vlm: VLMClient | None,
    max_steps: int,
    max_view_image_calls: int,
) -> Iterator[tuple[str, dict]]:
    yield "status", {"message": "starting agent loop"}

    is_freeform = eval_mode == "freeform"
    system_prompt = SYSTEM_PROMPT + (
        SYSTEM_PROMPT_FREEFORM_SUFFIX if is_freeform else SYSTEM_PROMPT_MCQ_SUFFIX
    )
    user_prompt = _format_question(question, choices, eval_mode)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    vi_used = 0
    step = 0

    while step < max_steps:
        budget = _budget_line(step + 1, max_steps, vi_used, max_view_image_calls)
        messages.append({"role": "user", "content": budget})

        assistant_msg = llm.chat(messages, tools=TOOLS)
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            yield "answer", {
                "response": (assistant_msg.get("content") or "").strip(),
                "steps": step,
            }
            return

        step += 1
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            thought = fn_args.get("thought", "")
            if thought:
                yield "thought", {"step": step, "text": thought}
            yield "tool_call", {"step": step, "tool": fn_name, "args": fn_args}

            if fn_name == "view_image" and vi_used >= max_view_image_calls:
                obs = "[view_image] budget reached; answer from prior views."
            else:
                result = _execute_tool(fn_name, fn_args, memory_dir, vlm)
                if fn_name == "view_image":
                    vi_used += 1
                obs = format_observation(fn_name, fn_args, result)

            yield "tool_result", {"step": step, "tool": fn_name, "observation": obs}
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": obs,
            })

    messages.append({
        "role": "user",
        "content": (
            f"You have used all {max_steps} steps. Stop calling tools and "
            "write your final answer now as plain text."
        ),
    })
    final_msg = llm.chat(messages, tools=None)
    yield "answer", {
        "response": (final_msg.get("content") or "").strip(),
        "steps": step,
    }


# ── MCQ helper ───────────────────────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_mcq_letter(
    final_text: str,
    letters: list[str],
    choices: dict[str, str] | None = None,
) -> str:
    """Extract the chosen letter from a free-text MCQ answer.

    Strategy (in order):
      1. Explicit "Answer: <L>. <text>" — cross-check L vs text.
      2. Verbatim option text anywhere in the response → its letter.
      3. First standalone letter token.
      4. Fallback to letters[0].
    """
    if not final_text:
        return letters[0]

    norm_choices = (
        {L: (t or "").strip().lower() for L, t in choices.items()}
        if choices else {}
    )

    def _letter_for_text(blob: str) -> str | None:
        if not norm_choices:
            return None
        b = blob.lower()
        hits = [L for L, t in norm_choices.items() if t and t in b]
        hits.sort(key=lambda L: len(norm_choices[L]), reverse=True)
        return hits[0] if hits else None

    m = re.search(r"(?i)\banswer\s*[:\-]\s*([A-D])\b\s*\.?\s*(.*)", final_text)
    if m:
        letter = m.group(1).upper()
        tail = m.group(2).split("\n")[0].strip()
        text_letter = _letter_for_text(tail) if tail else None
        if text_letter and text_letter != letter and letter in letters:
            return text_letter
        if letter in letters:
            return letter

    text_letter = _letter_for_text(final_text)
    if text_letter:
        return text_letter

    for tok in _LETTER_RE.findall(final_text):
        if tok in letters:
            return tok
    return letters[0]

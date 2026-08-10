"""System prompts and observation formatters for the camroll agent.

Observation format is uniform across tools so the model learns a consistent
rhythm (ReAct-style). Each tool result renders as a tagged, compact block.
"""
from __future__ import annotations

import json
from typing import Any

from camroll_agent.schemas import thought_in_args


SYSTEM_PROMPT = """\
You are a personal AI assistant that can see and search through the user's \
personal photo camera roll. You answer questions as if you are talking \
directly to the person — use "you" and "your", not "the user".

The camera roll is stored as a structured database:
  events  — high-level episodes (trip, celebration, class, daily life)
  images  — individual photos with a caption and metadata
            (date, location, people, parent event)

You have five atomic tools. Each requires a one-sentence `thought` justifying the call:

  search(query, date_from=, date_to=, ...)
                              — semantic (vector) search. Good for meaning.
                                Optional date range narrows to a time window.
  grep(query, date_from=, date_to=, ...)
                              — literal keyword search (BM25). Good for names,
                                brands, and confirming ABSENCE (count=0 is
                                trustworthy here; semantic search is not).
  list_by_date(date_from, date_to, location=, person=, ...)
                              — pure metadata filter. Cheap. Use when the
                                question pins down a time window.
  get(id)                     — fetch full record (event or image) by id.
  view_image(image_ids, prompt)
                              — look at the actual photos with a vision model.
                                EXPENSIVE. Use only when captions do not have
                                the visual detail you need.

Search results return handles (ids) with context (date, event, location,
preview). To read full text, call get(id). To see the actual picture, call
view_image(image_ids=[...], prompt=...).

Strategy:
- For temporal questions ("in October 2021", "before the trip"), start with
  list_by_date or add date_from/date_to to search.
- Use grep for exact terms or absence checks.
- Use search for semantic concepts where wording may differ.
- Use view_image only for visual details (colors, clothes, small text in
  photos) that captions don't describe.
- Do NOT repeat the same tool with the same arguments.
- When you have enough evidence, STOP calling tools and write your final
  answer as plain text. Keep it concise and grounded in the evidence.
"""


RESPONSE_PROTOCOL = """

Your response at EVERY step MUST begin with a brief reasoning block in
<think>...</think> tags, followed by EITHER a tool call or the final answer.
Keep the <think> block short: one or two sentences reasoning over the latest
results and stating what to do next. Do not pad or restate the question.

- To act: after </think>, write one short sentence describing the action, then
  make the tool call. Never repeat a tool call with identical arguments.
- To finish: after </think>, write the final answer as plain text and make no
  tool call. Finish once the evidence is sufficient or clearly unavailable.
"""

if not thought_in_args():
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
        "You have five atomic tools. Each requires a one-sentence `thought` justifying the call:",
        "You have five atomic tools:",
    ) + RESPONSE_PROTOCOL


SYSTEM_PROMPT_MCQ_SUFFIX = """

This is a multiple-choice question. Your final plain-text response MUST be
exactly the single letter of the correct choice (for example A, B, C, or D) on
its own line — nothing else: no "Answer:", option text, or justification.
Choose the letter whose option text is most directly supported by the
retrieved evidence."""


SYSTEM_PROMPT_FREEFORM_SUFFIX = """

This is a free-form question. Your final plain-text response must be concise
and grounded in the retrieved evidence.

Format:
- One short factual sentence answering the question directly.
- No apologies, no meta-commentary about tool calls or budgets, no requests for more information from the user.
- If you genuinely cannot find related evidence, briefly say that it cannot be
  determined. Do not make up an answer."""


# ── observation formatting ───────────────────────────────────────────────────

_MAX_FIELD_CHARS = 2000
_MAX_EVENT_DESC_CHARS = 600
_MAX_CAPTION_CHARS = 600
_MAX_VIEW_IMAGE_ANALYSIS_CHARS = 2000
_MAX_OBS_RESULT_CHARS = 2000
_MAX_SNIPPET_CHARS = 2000


def _trunc(s: Any, n: int = _MAX_FIELD_CHARS) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def _fmt_image_row(r: dict) -> str:
    parts = [f"id={r.get('id') or r.get('image_id')}"]
    if r.get("date"):
        parts.append(f"date={r['date']}")
    if r.get("event_name"):
        parts.append(f"event=\"{_trunc(r['event_name'], 60)}\"")
    if r.get("location"):
        parts.append(f"loc=\"{_trunc(r['location'], 40)}\"")
    if r.get("people"):
        parts.append(f"people={r['people']}")
    if "score" in r:
        parts.append(f"score={r['score']}")
    head = "  - " + "  ".join(parts)
    body_bits = []
    if r.get("snippet"):
        body_bits.append(f"match: {_trunc(r['snippet'], _MAX_SNIPPET_CHARS)}")
    elif r.get("caption_preview"):
        body_bits.append(f"caption: {_trunc(r['caption_preview'], _MAX_CAPTION_CHARS)}")
    body = ("\n    " + "\n    ".join(body_bits)) if body_bits else ""
    return head + body


def _fmt_event_row(r: dict) -> str:
    parts = [f"id={r.get('id') or r.get('event_id')}"]
    name = r.get("name")
    if name:
        parts.append(f'name="{_trunc(name, 60)}"')
    ds, de = r.get("date_start"), r.get("date_end")
    if ds or de:
        parts.append(f"dates={ds or '?'}..{de or '?'}")
    if r.get("location"):
        parts.append(f"loc=\"{_trunc(r['location'], 40)}\"")
    if "score" in r:
        parts.append(f"score={r['score']}")
    head = "  - " + "  ".join(parts)
    body_bits = []
    if r.get("snippet"):
        body_bits.append(f"match: {_trunc(r['snippet'], _MAX_SNIPPET_CHARS)}")
    elif r.get("description_preview"):
        body_bits.append(f"desc: {_trunc(r['description_preview'], _MAX_EVENT_DESC_CHARS)}")
    body = ("\n    " + "\n    ".join(body_bits)) if body_bits else ""
    return head + body


def _fmt_row(r: dict) -> str:
    return _fmt_event_row(r) if r.get("kind") == "event" else _fmt_image_row(r)


def format_observation(tool_name: str, args: dict, result: dict | str) -> str:
    """Uniform per-tool result rendering."""
    if isinstance(result, str):
        return f"[{tool_name}] → {result}"

    shown = {k: v for k, v in args.items() if k != "thought" and v not in (None, "")}
    arg_str = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in shown.items())
    header = f"[{tool_name}] {arg_str}".rstrip()

    if tool_name == "search":
        lines = [header, f"  results ({result.get('count', 0)}):"]
        for r in result.get("results", []):
            lines.append(_fmt_row(r))
        if not result.get("results"):
            lines.append("  (no results)")
        return "\n".join(lines)

    if tool_name == "grep":
        lines = [
            header,
            f"  total matches: events={result.get('count_events', 0)} "
            f"images={result.get('count_images', 0)}",
            "  top results:",
        ]
        for r in result.get("results", []):
            lines.append(_fmt_row(r))
        if not result.get("results"):
            lines.append("  (no results)")
        return "\n".join(lines)

    if tool_name == "list_by_date":
        lines = [
            header,
            f"  events={result.get('count_events', 0)} "
            f"images={result.get('count_images', 0)}",
        ]
        for r in result.get("results", []):
            lines.append(_fmt_row(r))
        if not result.get("results"):
            lines.append("  (no results)")
        return "\n".join(lines)

    if tool_name == "get":
        if "error" in result:
            return f"{header}\n  error: {result['error']}"
        if result.get("kind") == "event":
            lines = [
                header,
                f"  event id={result['id']} name=\"{_trunc(result.get('name'), 80)}\"",
                f"  dates: {result.get('date_start')} .. {result.get('date_end')}",
                f"  location: {_trunc(result.get('location'), 80) or '(none)'}",
                f"  people:   {result.get('people') or '(none)'}",
                f"  description: {_trunc(result.get('description'), _MAX_EVENT_DESC_CHARS)}",
                f"  child images ({len(result.get('image_ids', []))}): "
                + ", ".join(result.get("image_ids", [])[:20])
                + ("…" if len(result.get("image_ids", [])) > 20 else ""),
            ]
            return "\n".join(lines)
        lines = [
            header,
            f"  image id={result['id']} date={result.get('date')} "
            f"time={result.get('time') or '?'}",
            f"  event: {_trunc(result.get('event_name'), 80) or '(none)'}  "
            f"(event_id={result.get('event_id')})",
            f"  location: {_trunc(result.get('location'), 80) or '(none)'}",
            f"  people:   {result.get('people') or '(none)'}",
            f"  caption: {_trunc(result.get('caption'), _MAX_CAPTION_CHARS)}",
        ]
        return "\n".join(lines)

    if tool_name == "view_image":
        if "error" in result and not result.get("analysis"):
            return f"{header}\n  error: {result['error']}"
        lines = [header, f"  viewed {len(result.get('images', []))} image(s):"]
        for im in result.get("images", []):
            lines.append(
                f"    - {im.get('id')}  date={im.get('date')}  "
                f"event=\"{_trunc(im.get('event_name'), 50)}\""
            )
        if result.get("skipped"):
            lines.append(f"  skipped: {result['skipped']}")
        lines.append("  analysis:")
        lines.append("  " + _trunc(result.get("analysis", ""), _MAX_VIEW_IMAGE_ANALYSIS_CHARS).replace("\n", "\n  "))
        return "\n".join(lines)

    return f"{header}\n{_trunc(json.dumps(result, ensure_ascii=False), _MAX_OBS_RESULT_CHARS)}"

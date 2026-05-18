"""System prompts and observation formatters for the camroll agent.

Observation format is uniform across tools so the model learns a consistent
rhythm (ReAct-style). Each tool result renders as a tagged, compact block.
"""
from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
You are a personal-memory assistant with tool access to a user's photo album.

The album is stored as a structured database:
  events  — high-level episodes (trip, celebration, class, daily life)
  images  — individual photos with a first-person caption and metadata
            (date, location, people, parent event)

You have five atomic tools. Each requires a one-sentence `thought`
justifying the call:

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

Soft-matching rule (IMPORTANT):
Captions rarely contain every word the user used in the question. If the
question mentions companions ("with friends", "with mom"), occasions, or
qualifiers, the matching event in the album may not include those exact
words. When the date / location / object / activity clearly matches,
PROCEED with that event as the answer. Do not reject the best-matching
event just because a person/qualifier is not literally mentioned in its
caption.

Answering rule (IMPORTANT):
Always commit to a concrete factual answer drawn from the strongest
evidence you retrieved. If you have plausible evidence (even imperfect),
answer with it. Reserve "I don't know" for cases where no event in the
album is plausibly related to the question. Never apologize, never mention
tool budgets, and never ask the user for permission to keep searching."""


SYSTEM_PROMPT_MCQ_SUFFIX = """

This task is multiple-choice. Your final plain-text response MUST follow
this exact format on a single line, then optional justification on the next:

    Answer: <LETTER>. <verbatim option text>
    Because: <one sentence citing evidence (event names, dates, ids)>

Critical rules to avoid letter/text mismatches:
1. Re-read the choices block carefully and copy the chosen option's text
   VERBATIM (character-for-character) from the list shown to you.
2. The LETTER must be the letter that labels that exact text in the choices
   block. Do NOT default to "A" — verify the mapping each time.
3. If you are uncertain, pick the option whose text is most directly
   supported by retrieved evidence; do not guess A unless A is supported."""


SYSTEM_PROMPT_FREEFORM_SUFFIX = """

This task is free-form. Your final plain-text response must be a concise
factual answer to the question, grounded in the retrieved evidence.

Format:
- One short factual sentence answering the question directly.
- No apologies, no meta-commentary about tool calls or budgets, no
  requests for more information from the user.
- If you genuinely cannot find any related evidence, say "I don't know"
  — but only as a last resort, not because of imperfect keyword matches."""


# ── observation formatting ───────────────────────────────────────────────────

_MAX_FIELD_CHARS = 180


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
        body_bits.append(f"match: {_trunc(r['snippet'], 500)}")
    elif r.get("caption_preview"):
        body_bits.append(f"caption: {_trunc(r['caption_preview'], 500)}")
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
        body_bits.append(f"match: {_trunc(r['snippet'], 500)}")
    elif r.get("description_preview"):
        body_bits.append(f"desc: {_trunc(r['description_preview'], 500)}")
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
                f"  description: {_trunc(result.get('description'), 500)}",
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
            f"  caption: {_trunc(result.get('caption'), 500)}",
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
        lines.append("  " + _trunc(result.get("analysis", ""), 2000).replace("\n", "\n  "))
        return "\n".join(lines)

    return f"{header}\n{_trunc(json.dumps(result, ensure_ascii=False), 1200)}"

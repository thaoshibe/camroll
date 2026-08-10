"""OpenAI-style tool schemas for the camroll-agent.

Every tool has a required `thought` field first (ReAct discipline: the model
must justify each call). Gemini's schema is derived from these at runtime
inside the corresponding LLM client.
"""
from __future__ import annotations

import copy
import os

THOUGHT_DESC = (
    "One short sentence stating why you are calling this tool now and what "
    "you expect to learn. Required before any action."
)

KIND_DESC = (
    "What to search. One of: 'both' (events and image captions), "
    "'events' (event descriptions only), 'captions' (image captions only)."
)

DATE_DESC = "ISO date YYYY-MM-DD. Optional."

def thought_in_args() -> bool:
    """Whether tool schemas expose the legacy ``thought`` argument."""
    return os.environ.get("CAMROLL_NO_THOUGHT_ARG", "0").lower() not in (
        "1", "true", "yes", "on",
    )


def _strip_thought_arg(tools: list[dict]) -> list[dict]:
    """Return tool schemas with reasoning moved out of function arguments."""
    cleaned = copy.deepcopy(tools)
    for tool in cleaned:
        params = tool.get("function", {}).get("parameters", {})
        params.get("properties", {}).pop("thought", None)
        if isinstance(params.get("required"), list):
            params["required"] = [
                name for name in params["required"] if name != "thought"
            ]
    return cleaned


_BASE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Semantic (vector) search over events and/or image captions. "
                "Good for finding things by meaning when you don't know the exact words. "
                "Returns handles (ids) with brief context (date, event, location, preview). "
                "To read the full text of a result, call get(id)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": THOUGHT_DESC},
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "top_k": {"type": "integer", "description": "Max results to return. Default 10."},
                    "kind": {"type": "string", "description": KIND_DESC},
                    "date_from": {"type": "string", "description": DATE_DESC},
                    "date_to":   {"type": "string", "description": DATE_DESC},
                },
                "required": ["thought", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Literal keyword search (BM25). Complements search for "
                "exact terms (names, brands, places) and for confirming ABSENCE "
                "of a term (count=0 is trustworthy here, unlike semantic search "
                "which always returns something)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": THOUGHT_DESC},
                    "query": {"type": "string",
                              "description": "Literal terms. All terms must appear in a match (implicit AND)."},
                    "kind": {"type": "string", "description": KIND_DESC},
                    "top_k": {"type": "integer", "description": "Max results to return. Default 10."},
                    "date_from": {"type": "string", "description": DATE_DESC},
                    "date_to":   {"type": "string", "description": DATE_DESC},
                },
                "required": ["thought", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_by_date",
            "description": (
                "List events and/or images filtered by metadata (date range, "
                "location, person). Pure structured filter — cheap, no embedding. "
                "Use when the question pins down a time window like "
                "'late October 2021' or a known location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": THOUGHT_DESC},
                    "date_from": {"type": "string", "description": DATE_DESC},
                    "date_to":   {"type": "string", "description": DATE_DESC},
                    "kind": {"type": "string", "description": KIND_DESC},
                    "location": {"type": "string", "description": "Substring match. Optional."},
                    "person":   {"type": "string", "description": "Substring match. Optional."},
                    "limit":    {"type": "integer", "description": "Max rows to return. Default 50."},
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get",
            "description": (
                "Fetch the full record for one event (id starts with 'ev_') or "
                "one image (id starts with 'img_'). Use this after a search "
                "when you want the full text instead of the preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": THOUGHT_DESC},
                    "id": {"type": "string",
                           "description": "An event_id ('ev_...') or image_id ('img_...')."},
                },
                "required": ["thought", "id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": (
                "Look at one or more actual photos with a vision model, using "
                "a prompt you write yourself. This is the EXPENSIVE tool — use "
                "it only when captions/descriptions don't answer the question. "
                "Pass a list of image_ids (1-6). All images are analyzed "
                "together in one call, so you can ask comparison questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": THOUGHT_DESC},
                    "image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of image_ids (e.g. ['img_abc', 'img_def']). "
                                       "Up to 6 images per call.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What do you want to know about these photos? Be specific.",
                    },
                },
                "required": ["thought", "image_ids", "prompt"],
            },
        },
    },
]

TOOLS: list[dict] = (
    _BASE_TOOLS if thought_in_args() else _strip_thought_arg(_BASE_TOOLS)
)

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]

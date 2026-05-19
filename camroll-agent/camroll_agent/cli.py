"""Command-line interface.

Three subcommands:

  camroll-agent inspect  conversation.json
      Print summary of a conversation JSON (image count, date range).

  camroll-agent build    conversation.json -o memory/
      Stage 1: caption every photo + group events with a VLM.

  camroll-agent index    memory/
      Stage 2: build SQLite + FTS5 + vector store on top of Stage 1 output.

  camroll-agent ask      "When did I go to the lake?"  --memory memory/
      Stage 3: run the ReAct agent.

  camroll-agent run      conversation.json -o memory/
      Convenience: build + index in one shot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from camroll_agent import build_memory, index
from camroll_agent.agent import Agent
from camroll_agent.llm import build_vlm


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = build_memory.inspect(args.spec)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    summary = build_memory.run(
        args.spec,
        output_dir=args.output_dir,
        backend=args.vlm_backend,
        model=args.vlm_model,
        max_images=args.max_images,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    summary = index.run(
        args.memory_dir,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    build_summary = build_memory.run(
        args.spec,
        output_dir=args.output_dir,
        backend=args.vlm_backend,
        model=args.vlm_model,
        max_images=args.max_images,
        resume=args.resume,
    )
    index_summary = index.run(
        build_summary["output_dir"],
        embedding_model=args.embedding_model,
    )
    print(json.dumps(
        {"build": build_summary, "index": index_summary},
        indent=2, ensure_ascii=False,
    ))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    vlm_backend = args.vlm_backend if args.vlm_backend != "openai" else args.llm_backend
    vlm = build_vlm(vlm_backend, args.vlm_model)
    agent = Agent(
        memory_dir=args.memory_dir,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        vlm=vlm,
        max_steps=args.max_steps,
        max_view_image_calls=args.max_view_image_calls,
    )
    _fmt_memory_header(
        args.memory_dir, args.llm_backend, args.llm_model,
        vlm_backend=vlm_backend, vlm_model=args.vlm_model,
        max_steps=args.max_steps, max_view_image_calls=args.max_view_image_calls,
    )

    if args.stream:
        step = 0
        max_steps = args.max_steps
        viewed_ids: list[str] = []
        retrieved_ids: list[str] = []
        for evt, data in agent.ask_streaming(args.question):
            if evt == "status":
                _fmt_status(data["message"])
            elif evt == "thought":
                _fmt_thought(data["text"])
            elif evt == "tool_call":
                step += 1
                _fmt_tool_call(data["tool"], data["args"], step, max_steps)
                if data["tool"] == "view_image":
                    for iid in (data["args"].get("image_ids") or []):
                        if iid not in viewed_ids:
                            viewed_ids.append(iid)
            elif evt == "tool_result":
                _fmt_tool_result(data["tool"], data["observation"])
                if data["tool"] in ("grep", "list_by_date", "get"):
                    for iid in _IMG_ID_RE.findall(data["observation"] or ""):
                        if iid not in retrieved_ids:
                            retrieved_ids.append(iid)
            elif evt == "answer":
                _fmt_answer(data["response"], steps_used=step, max_steps=max_steps)
        image_ids = viewed_ids if viewed_ids else retrieved_ids
        _fmt_relevant_images(args.memory_dir, image_ids, viewed=bool(viewed_ids))
        return 0

    result = agent.ask(args.question)
    if args.json:
        print(json.dumps({
            "answer": result.final_text,
            "steps": result.steps,
            "view_image_calls": result.view_image_calls,
            "latency_s": result.latency_s,
            "stopped_reason": result.stopped_reason,
        }, indent=2, ensure_ascii=False))
    else:
        _fmt_answer(result.final_text)
    return 0


# ── pretty-print helpers ──────────────────────────────────────────────────────

import os as _os
import re as _re
_IMG_ID_RE = _re.compile(r'\bimg_[0-9a-f]+\b')
_TTY   = _os.isatty(1)
_R     = "\033[0m"   if _TTY else ""
_BOLD  = "\033[1m"   if _TTY else ""
_DIM   = "\033[2m"   if _TTY else ""
_CYAN  = "\033[36m"  if _TTY else ""
_GREEN = "\033[32m"  if _TTY else ""
_GRAY  = "\033[90m"  if _TTY else ""
_YELLOW= "\033[33m"  if _TTY else ""

_TOOL_LABEL = {
    "search":       "Searching memories",
    "grep":         "Keyword search",
    "list_by_date": "Listing by date",
    "get":          "Retrieving record",
    "view_image":   "Viewing image",
}


def _fmt_memory_header(
    memory_dir: str,
    llm_backend: str,
    llm_model: str | None,
    *,
    vlm_backend: str = "openai",
    vlm_model: str | None = None,
    max_steps: int = 25,
    max_view_image_calls: int = 5,
    view_image: bool = False,
) -> None:
    from camroll_agent import store
    try:
        conn = store.connect(memory_dir, read_only=True)
        s = store.stats(conn)
        conn.close()
        date_range = f"{s['date_earliest']} → {s['date_latest']}" if s["date_earliest"] else "no dates"
        width = min(_os.get_terminal_size().columns if _TTY else 60, 60)
        print(f"{_GRAY}{'─' * width}{_R}")
        print(f"  {_BOLD}Memory{_R}  "
              f"{_CYAN}{s['n_events']} events  {s['n_images']} images{_R}  "
              f"{_GRAY}{date_range}{_R}")
        print(f"  {_GRAY}Agent {_R}  {llm_backend} / {llm_model or 'default'}")
        print(f"  {_GRAY}Vision{_R}  {vlm_backend} / {vlm_model or 'default'}")
        print(f"  {_GRAY}Budget{_R}  "
              f"max {max_steps} steps  "
              f"max {max_view_image_calls} view_image calls")
        print(f"{_GRAY}{'─' * width}{_R}\n")
    except Exception:
        pass


def _fmt_status(msg: str) -> None:
    print(f"{_DIM}  ⋯  {msg}{_R}", flush=True)


def _fmt_thought(text: str) -> None:
    short = text if len(text) <= 90 else text[:90] + "…"
    print(f"{_GRAY}  ╎  {short}{_R}", flush=True)


def _fmt_tool_call(tool: str, tool_args: dict, step: int, max_steps: int) -> None:
    label = _TOOL_LABEL.get(tool, tool)
    budget = f"{_GRAY}[{step}/{max_steps}]{_R}"
    # collect all meaningful args (skip thought)
    parts = []
    for field in ("query", "id", "date_from", "date_to", "image_ids"):
        val = tool_args.get(field)
        if val not in (None, "", []):
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val[:3])
            parts.append(f'{_DIM}{field}{_R}={_YELLOW}"{val}"{_R}')
    # also show thought as context
    thought = tool_args.get("thought", "")
    if thought:
        short_thought = thought if len(thought) <= 80 else thought[:80] + "…"
        print(f"{_GRAY}  ╎  {short_thought}{_R}", flush=True)
    arg_str = "  " + "  ".join(parts) if parts else ""
    print(f"\n{_CYAN}  ◆  {label}{_R}  {budget}{arg_str}", flush=True)


def _fmt_tool_result(tool: str, observation: str) -> None:
    if not observation:
        return
    lines = [ln.strip() for ln in observation.splitlines() if ln.strip()]
    # show up to 4 lines of the result for full context
    shown = []
    for line in lines[:6]:
        if line.startswith("["):          # skip the header line echoing args
            continue
        shown.append(line)
        if len(shown) >= 4:
            break
    if not shown and lines:
        shown = [lines[0]]
    for i, line in enumerate(shown):
        short = line if len(line) <= 110 else line[:110] + "…"
        prefix = f"{_GREEN}     ✓  {_R}" if i == 0 else "        "
        print(f"{prefix}{short}", flush=True)


def _fmt_answer(text: str, steps_used: int = 0, max_steps: int = 0) -> None:
    width = min(_os.get_terminal_size().columns if _TTY else 60, 60)
    step_info = f"  {_GRAY}{steps_used} tool call{'s' if steps_used != 1 else ''}{_R}" if steps_used else ""
    print(f"\n{_GRAY}{'─' * width}{_R}{step_info}\n")
    print(f"{_BOLD}{text}{_R}\n", flush=True)


def _fmt_relevant_images(memory_dir: str, image_ids: list[str], *, viewed: bool = False) -> None:
    if not image_ids:
        return
    try:
        from camroll_agent import store
        conn = store.connect(memory_dir, read_only=True)
        rows = []
        for iid in image_ids[:8]:
            row = store.get_image(conn, iid)
            if row and row.get("path"):
                rows.append((row["path"], row.get("date", "")))
        conn.close()
        if not rows:
            return
        label = "image(s) viewed by agent" if viewed else "image(s) from retrieved captions"
        print(f"{_GRAY}   │{_R}")
        print(f"{_GRAY}   │  {label}{_R}")
        for path, date in rows:
            print(f"{_GRAY}   └─ {date}  {path}{_R}")
        print()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="camroll-agent",
        description="Agentic search over a personal photo library.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Summarize a conversation JSON.")
    p_inspect.add_argument("spec", help="Path to conversation JSON.")
    p_inspect.set_defaults(func=_cmd_inspect)

    # build
    p_build = sub.add_parser("build", help="Stage 1: VLM captioning + event grouping.")
    p_build.add_argument("spec", help="Path to conversation JSON.")
    p_build.add_argument("-o", "--output-dir", required=True,
                         help="Directory to write events.json / images.json.")
    p_build.add_argument("--vlm-backend", default="openai",
                         choices=["openai", "gemini", "local"])
    p_build.add_argument("--vlm-model", default=None,
                         help="Model name (e.g. gpt-4o, gemini-2.5-flash).")
    p_build.add_argument("--max-images", type=int, default=None,
                         help="Process at most N images (useful for smoke tests).")
    p_build.add_argument("--resume", action="store_true",
                         help="Continue an interrupted run in output_dir.")
    p_build.set_defaults(func=_cmd_build)

    # index
    p_index = sub.add_parser("index", help="Stage 2: build SQLite + vector store.")
    p_index.add_argument("memory_dir",
                         help="Directory containing events.json / images.json.")
    p_index.add_argument("--embedding-model",
                         default=index.DEFAULT_EMBEDDING_MODEL,
                         help="Embedding model (sentence-transformers ID or OpenAI model name).")
    p_index.set_defaults(func=_cmd_index)

    # run = build + index
    p_run = sub.add_parser("run", help="Stage 1 + Stage 2 in one shot.")
    p_run.add_argument("spec", help="Path to conversation JSON.")
    p_run.add_argument("-o", "--output-dir", required=True)
    p_run.add_argument("--vlm-backend", default="openai",
                       choices=["openai", "gemini", "local"])
    p_run.add_argument("--vlm-model", default=None)
    p_run.add_argument("--max-images", type=int, default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--embedding-model",
                       default=index.DEFAULT_EMBEDDING_MODEL)
    p_run.set_defaults(func=_cmd_run)

    # ask
    p_ask = sub.add_parser("ask", help="Stage 3: run the agent against an indexed memory.")
    p_ask.add_argument("question")
    p_ask.add_argument("--memory", "--memory-dir", dest="memory_dir", required=True,
                       help="Directory built by `camroll-agent index`.")
    p_ask.add_argument("--llm-backend", default="openai",
                       choices=["openai", "gemini", "local"])
    p_ask.add_argument("--llm-model", default=None)
    p_ask.add_argument("--vlm-backend", default="openai",
                       choices=["openai", "gemini", "local"],
                       help="VLM backend for view_image.")
    p_ask.add_argument("--vlm-model", default=None)
    p_ask.add_argument("--max-steps", type=int, default=25)
    p_ask.add_argument("--max-view-image-calls", type=int, default=5)
    p_ask.add_argument("--no-stream", dest="stream", action="store_false",
                       help="Suppress live tool-call output, print final answer only.")
    p_ask.set_defaults(stream=True)
    p_ask.add_argument("--json", action="store_true",
                       help="Output a JSON object instead of plain text.")
    p_ask.set_defaults(func=_cmd_ask)

    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())

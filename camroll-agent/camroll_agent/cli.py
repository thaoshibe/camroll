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
    vlm = None
    if args.enable_view_image:
        vlm = build_vlm(args.vlm_backend, args.vlm_model)
    agent = Agent(
        memory_dir=args.memory_dir,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        vlm=vlm,
        max_steps=args.max_steps,
        max_view_image_calls=args.max_view_image_calls,
    )
    if args.stream:
        for evt, data in agent.ask_streaming(args.question):
            if evt == "status":
                print(f"· {data['message']}", flush=True)
            elif evt == "thought":
                print(f"[thought {data['step']}] {data['text']}", flush=True)
            elif evt == "tool_call":
                args_brief = {k: v for k, v in data["args"].items() if v not in (None, "")}
                print(f"[tool_call] {data['tool']}({args_brief})", flush=True)
            elif evt == "tool_result":
                first = data["observation"].splitlines()[0] if data["observation"] else ""
                print(f"[tool_result] {first}", flush=True)
            elif evt == "answer":
                print()
                print(data["response"])
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
        print(result.final_text)
    return 0


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
                       choices=["openai", "gemini"])
    p_ask.add_argument("--llm-model", default=None)
    p_ask.add_argument("--vlm-backend", default="openai",
                       choices=["openai", "gemini", "local"],
                       help="Used for view_image when --enable-view-image is set.")
    p_ask.add_argument("--vlm-model", default=None)
    p_ask.add_argument("--enable-view-image", action="store_true",
                       help="Allow the agent to call view_image (requires VLM).")
    p_ask.add_argument("--max-steps", type=int, default=25)
    p_ask.add_argument("--max-view-image-calls", type=int, default=5)
    p_ask.add_argument("--stream", action="store_true",
                       help="Print thoughts / tool calls / results as they happen.")
    p_ask.add_argument("--json", action="store_true",
                       help="Output a JSON object instead of plain text.")
    p_ask.set_defaults(func=_cmd_ask)

    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())

"""camroll-agent — agentic search over a personal photo library.

Three stages:
  1. build_memory   VLM captions each photo + groups them into events.
  2. index          builds a SQLite+FTS5 store and a vector index for fast search.
  3. agent          a ReAct loop with 5 atomic tools (search, grep,
                    list_by_date, get, view_image) answers questions about the
                    photo library.

Quickstart:

    from camroll_agent import build_memory, index, Agent

    # 1. caption + group events (one-time per album)
    build_memory.run("my_conversation.json", output_dir="memory/")

    # 2. build the searchable index
    index.run("memory/")

    # 3. ask questions
    agent = Agent(memory_dir="memory/")
    print(agent.ask("When did I go to Lake Michigan?"))
"""
from __future__ import annotations

__version__ = "0.1.0"

from camroll_agent.agent import Agent

__all__ = ["Agent", "__version__"]

"""End-to-end quickstart for camroll-agent.

Runs the full 3-stage pipeline on `sample_conversation.json` (6 real
photos from 2 events — autumn lake day in 2005 and summer at the lake in
2006), then asks a question about the album.

Before running:
  export OPENAI_API_KEY=...

From the repo root:
  python examples/quickstart.py
"""
from pathlib import Path

from camroll_agent import Agent, build_memory, index

HERE = Path(__file__).resolve().parent
SPEC = HERE / "sample_conversation.json"
MEMORY_DIR = HERE / "memory"

print(f"[1/3] Stage 1: VLM captions + event grouping  (output → {MEMORY_DIR})")
build_summary = build_memory.run(
    SPEC,
    output_dir=MEMORY_DIR,
    backend="openai",
)
print(
    f"      events={build_summary['n_events']}  images={build_summary['n_images']}"
)

print(f"\n[2/3] Stage 2: building SQLite + vector store")
index_summary = index.run(MEMORY_DIR)
print(
    f"      sqlite: {index_summary['sqlite']}  "
    f"vector backend: {index_summary['vector']['backend']}"
)

print(f"\n[3/3] Stage 3: ask the agent")
agent = Agent(memory_dir=str(MEMORY_DIR), llm_backend="openai")
question = "When did this user spend time at the lake, and what season was it?"
print(f"      Q: {question}")
result = agent.ask(question)
print(f"      A: {result.final_text}")
print(
    f"      (steps={result.steps}  view_image={result.view_image_calls}  "
    f"latency={result.latency_s}s)"
)

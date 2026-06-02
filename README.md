<p align="center">
  <img src="assets/banner.svg" alt="camroll — Personal AI Agent for Camera Roll VQA" width="100%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://thaoshibe.github.io/camroll/"><img src="https://img.shields.io/badge/Project-Page-2a6310.svg?logo=github&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/thaoshibe/camroll-agent"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Demo-Spaces-ffd21e.svg" alt="Demo"></a>
  <a href="page/demo.html"><img src="https://img.shields.io/badge/🚀_Demo-Live-ff6b35.svg" alt="Live Demo"></a>
</p>

> **TL;DR:** `camroll-agent` is an **AI agent** that does VQA on a personal camera roll.
> 1. index your camera roll into a hierarchical queryable memory (events << captions << images).
> 2. the agent answers questions over that memory using 5 atomic tools: `search`, `grep`, `list_by_date`, `get`, and `view_image`.

---

## Install

<table>
<tr>
<th width="50%">🌐 Use API <sup>(OpenAI, Gemini)</sup></th>
<th width="50%">💻 Local <sup>(GPU required)</sup></th>
</tr>
<tr>
<td valign="top">

```bash
git clone https://github.com/thaoshibe/camroll
cd camroll

conda create -n camroll python=3.10 -y
conda activate camroll

pip install -r requirements.txt
pip install -e camroll-agent/
```

OpenAI + Gemini APIs. No torch (~50 MB install).

</td>
<td valign="top">

```bash
git clone https://github.com/thaoshibe/camroll
cd camroll

conda create -n camroll-local python=3.10 -y
conda activate camroll-local

pip install -r requirements_local.txt
pip install -e camroll-agent/
```

Adds Qwen-VL / Kimi-VL + `sentence-transformers`. Needs CUDA (~3 GB install).

</td>
</tr>
</table>

Set the API key for whichever cloud backend you use:

```bash
export OPENAI_API_KEY=sk-…       # for OpenAI VLM/LLM + embeddings (default)
export GEMINI_API_KEY=…           # for Gemini VLM/LLM
```

The default embedding model is **OpenAI's `text-embedding-3-small`** (fast,
no local install). If you'd rather use a local sentence-transformers model
(free, offline), install `requirements_local.txt` and pass
`--embedding-model sentence-transformers/all-MiniLM-L6-v2` at index time.

## Quickstart

All commands below assume you are at the **repo root** (`camroll/`).

### 1. Prepare a conversation JSON

```jsonc
// my_album.json
{
  "root_folder": "/absolute/path/to/photos",   // all image paths resolve relative to this
  "profile_image": "profile.jpg",              // reference photo of the person (used for identity context)
  "library_description": "This is my personal photo camera roll.",
  "turns": [
    {"date": "2005-10-01", "user": {"image": "847410131.jpg"}},
    {"date": "2005-10-01", "user": {"image": "847410831.jpg"}},
    {"date": "2005-10-15", "user": {"image": "851200001.jpg"}}
  ]
}
```

A ready-to-run sample (6 real photos) is at `camroll-agent/examples/sample_conversation.json`.

Preview what will be processed without calling any API:

```bash
python -m camroll_agent inspect camroll-agent/examples/sample_conversation.json
```

### 2. Build the memory (Stage 1 + Stage 2)

**Using OpenAI** (default):

```bash
export OPENAI_API_KEY=sk-…

python -m camroll_agent run camroll-agent/examples/sample_conversation.json -o memory/ \
    --vlm-backend openai \
    --vlm-model gpt-4o \
    --embedding-model text-embedding-3-small
```

**Using Gemini**:

```bash
export GEMINI_API_KEY=…

python -m camroll_agent run camroll-agent/examples/sample_conversation.json -o memory/ \
    --vlm-backend gemini \
    --vlm-model gemini-2.5-flash \
    --embedding-model text-embedding-3-small   # still needs OPENAI_API_KEY unless you use local embeddings
```

**Fully local — no API key needed (GPU required)**:

```bash
python -m camroll_agent run camroll-agent/examples/sample_conversation.json -o memory/ \
    --vlm-backend local \
    --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

> First run downloads Qwen2.5-VL-7B from HuggingFace (~15 GB). Cached after that.

**All `run` flags:**

| Flag | Default | Description |
|---|---|---|
| `-o / --output-dir` | *(required)* | Where to write the memory |
| `--vlm-backend` | `openai` | `openai` \| `gemini` \| `local` |
| `--vlm-model` | backend default | e.g. `gpt-4o`, `gemini-2.5-flash`, `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--embedding-model` | `text-embedding-3-small` | OpenAI model name or any `sentence-transformers` ID |
| `--max-images` | all | Process at most N images (useful for smoke tests) |
| `--resume` | off | Continue an interrupted run |

Or run Stage 1 and Stage 2 separately:

```bash
python -m camroll_agent build camroll-agent/examples/sample_conversation.json -o memory/  \
    --vlm-backend openai --vlm-model gpt-4o --max-images 10 --resume

python -m camroll_agent index memory/ \
    --embedding-model text-embedding-3-small
```

### 3. Ask questions

**Using OpenAI** (default):

```bash
export OPENAI_API_KEY=sk-…

python -m camroll_agent ask "When did I go to Lake Michigan?" \
    --memory memory/ \
    --llm-backend openai \
    --llm-model gpt-4o
```

**Using Gemini**:

```bash
export GEMINI_API_KEY=…

python -m camroll_agent ask "When did I go to Lake Michigan?" \
    --memory memory/ \
    --llm-backend gemini \
    --llm-model gemini-2.5-flash
```

**Fully local (no API key needed, GPU required)**:

```bash
python -m camroll_agent ask "When did I go to Lake Michigan?" \
    --memory memory/ \
    --llm-backend local \
    --llm-model Qwen/Qwen2.5-Coder-7B-Instruct
```

**All `ask` flags:**

| Flag | Default | Description |
|---|---|---|
| `--memory` | *(required)* | Memory directory built in Step 2 |
| `--llm-backend` | `openai` | `openai` \| `gemini` \| `local` (Qwen, GPU required) |
| `--llm-model` | backend default | e.g. `gpt-4o`, `gemini-2.5-flash`, `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--vlm-backend` | `openai` | VLM used by `view_image` (`openai` \| `gemini` \| `local`) |
| `--vlm-model` | backend default | e.g. `gpt-4o`, `gemini-2.5-flash`, `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--no-stream` | off | Suppress live tool output, print final answer only |
| `--json` | off | Output full JSON (answer + tool trace + latency) |
| `--max-steps` | `25` | Max ReAct steps before stopping |
| `--max-view-image-calls` | `5` | Cap on expensive `view_image` calls |

Examples:

```bash
# use a different VLM for viewing photos (default: openai)
python -m camroll_agent ask "What color was the car at the airport?" \
    --memory memory/ --vlm-backend local

# get structured JSON output (answer + tool trace + latency)
python -m camroll_agent ask "When did I go to Lake Michigan?" \
    --memory memory/ --json

# suppress live output, print final answer only
python -m camroll_agent ask "When did I go to Lake Michigan?" \
    --memory memory/ --no-stream
```

## Python API

```python
from camroll_agent import build_memory, index, Agent

build_memory.run("my_album.json", output_dir="memory/", backend="openai")
index.run("memory/")

agent = Agent(memory_dir="memory/", llm_backend="openai")
result = agent.ask("When did I go to Lake Michigan?")
print(result.final_text)
print(result.tool_trace)
```

Streaming:

```python
for evt, data in agent.ask_streaming("..."):
    print(evt, data)
```

## The 5 atomic tools

The agent reasons over 5 deliberately small, single-purpose tools:

| Tool | What it does | Cost |
|---|---|---|
| `search(query, …)` | Semantic (vector) search over events + captions | cheap |
| `grep(query, …)` | Literal BM25 keyword search via SQLite FTS5 | cheap |
| `list_by_date(date_from, date_to, …)` | Pure metadata filter | cheap |
| `get(id)` | Fetch the full event or image record by id | cheap |
| `view_image(image_ids, prompt)` | Look at the actual photos with a VLM | expensive |

Every tool requires a one-sentence `thought` argument before it can be
called — this is the ReAct discipline. The agent terminates by emitting
plain text (no `answer` tool).

## Customizing

### Swap the LLM

Any class that implements `LLMClient.chat(messages, tools)` works:

```python
from camroll_agent.llm.base import LLMClient
from camroll_agent import Agent

class MyLLM(LLMClient):
    def chat(self, messages, tools=None, *, tool_choice="auto"):
        # return an OpenAI-shaped assistant message dict
        ...

agent = Agent(memory_dir="memory/", llm=MyLLM())
```

### Swap the VLM (for Stage 1 captioning and view_image)

```python
from camroll_agent.llm.base import VLMClient
from camroll_agent import build_memory

class MyVLM(VLMClient):
    def generate(self, prompt: str, image_paths: list[str]) -> str:
        ...

build_memory.run("my_album.json", output_dir="memory/", vlm=MyVLM())
```

### Swap embeddings

```python
from camroll_agent import index
from camroll_agent.vector import EmbeddingClient

class MyEmbed:
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...

index.run("memory/", embedding_client=MyEmbed())
```

## Package layout

```
camroll-agent/
├── pyproject.toml
├── camroll_agent/
│   ├── __init__.py
│   ├── build_memory.py    Stage 1: VLM captioning + event grouping
│   ├── index.py           Stage 2: SQLite + FTS5 + vector store
│   ├── store.py             ↳ SQLite schema + read/write helpers
│   ├── vector.py            ↳ embeddings + FAISS / numpy
│   ├── agent.py           Stage 3: ReAct loop, pluggable backends
│   ├── tools.py             ↳ the 5 atomic tools
│   ├── prompts.py           ↳ system prompts + observation formatter
│   ├── schemas.py           ↳ OpenAI-style tool schemas
│   ├── cli.py             `camroll-agent inspect/build/index/run/ask`
│   └── llm/               pluggable VLM + LLM backends
│       ├── base.py
│       ├── openai_client.py
│       ├── gemini_client.py
│       └── local_client.py
└── examples/
    ├── sample_conversation.json
    └── quickstart.py
```

---

## Citation

If this code helps your research, please cite the camroll paper.

## License

MIT.

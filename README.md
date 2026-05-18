<p align="center">
  <img src="assets/banner.svg" alt="camroll — Personal Camera Roll Visual Question Answering" width="100%">
</p>

# camroll

An **agentic search engine over a personal photo library**. Given a JSON of
your photos (paths + dates + chat/metadata), `camroll`:

1. **Captions** every photo with a vision model and **groups** them into
   coherent events (a trip, a hangout, a class activity, …).
2. **Indexes** captions and events into a SQLite + FTS5 keyword store and a
   vector store for semantic search.
3. **Answers** natural-language questions about your library with a ReAct
   agent that has 5 atomic search tools.

This repo also includes a static demo website (`page/`, `index.html`,
`yfcc_users/`) hosted on GitHub Pages — it lets you browse 14 YFCC100M
users' photo rolls with pre-recorded agent traces, no install required.
Everything below is about the Python package in `camroll-agent/`.

---

## Install

Pick whichever VLM/LLM backend you want, and add `[embeddings]` for the
default local sentence-transformers embedding model:

```bash
pip install camroll-agent[openai,embeddings]      # OpenAI + local embeddings
pip install camroll-agent[gemini,embeddings]      # Gemini + local embeddings
pip install camroll-agent[anthropic,embeddings]   # Claude + local embeddings
pip install camroll-agent[local,embeddings]       # local HF VLM (needs GPU) + embeddings
pip install camroll-agent[all]                     # everything except [local]
```

Set the API key for whichever cloud backend you use:

```bash
export OPENAI_API_KEY=sk-…
export GEMINI_API_KEY=…
export ANTHROPIC_API_KEY=sk-ant-…
```

If you'd rather use **OpenAI embeddings** (faster, no torch download), you
can skip `[embeddings]` and pass `--embedding-model text-embedding-3-small`
at index time. You'll get a clear ImportError otherwise:

```
ImportError: sentence-transformers is required for the default embedding
model 'sentence-transformers/all-MiniLM-L6-v2'. Install it with:
    pip install camroll-agent[embeddings]
or pick a different embedding model (e.g. --embedding-model text-embedding-3-small to use OpenAI).
```

## Quickstart

### 1. Prepare a conversation JSON

```jsonc
// my_album.json
{
  "root_folder": "/absolute/path/to/photos",
  "profile_image": "profile.jpg",
  "library_description": "A 2005-2013 album from a college student.",
  "turns": [
    {"date": "2005-10-01", "user": {"image": "847410131.jpg"}},
    {"date": "2005-10-01", "user": {"image": "847410831.jpg"}},
    {"date": "2005-10-15", "user": {"image": "851200001.jpg"}}
  ]
}
```

Each turn has a `date` and a `user.image` path (absolute or relative to
`root_folder`). You can include extra fields on the turn — they'll be
passed to the VLM as additional context.

### 2. Build the memory (Stage 1 + Stage 2)

```bash
camroll-agent run my_album.json -o memory/
```

Or step by step:

```bash
camroll-agent build my_album.json -o memory/    # VLM captioning + event grouping
camroll-agent index memory/                     # SQLite + vector store
```

You can preview what would be processed without calling the VLM:

```bash
camroll-agent inspect my_album.json
```

### 3. Ask questions

```bash
camroll-agent ask "When did I go to Lake Michigan?" --memory memory/
```

For a streaming trace of thoughts + tool calls:

```bash
camroll-agent ask "..." --memory memory/ --stream
```

To let the agent actually look at photos with a VLM (for visual details
that captions miss):

```bash
camroll-agent ask "What color was the car at the airport?" \
    --memory memory/ --enable-view-image
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
| `search_memory(query, …)` | Semantic (vector) search over events + captions | cheap |
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
│       ├── anthropic_client.py
│       └── local_client.py
└── examples/
    ├── sample_conversation.json
    └── quickstart.py
```

---

## Citation

If this code helps your research, please cite the camroll / kii paper(s).

## License

MIT.

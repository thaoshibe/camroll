"""Vector index for semantic search over events and image captions.

Embedding backends:
  - OpenAI text-embedding-3-small (DEFAULT, requires OPENAI_API_KEY)
  - sentence-transformers (local, free, ~80MB download on first use —
    install with requirements_local.txt)

If neither dependency is installed, build/query raise ImportError with an
explicit install hint. You can also bring your own by passing a custom
`EmbeddingClient` to `build_vector_index(...)`.

Storage:
  vector_store/
    records.json     # list of indexed records (id, text, payload)
    manifest.json    # {embedding_model, dimension, backend}
    memory.index     # FAISS index (if faiss-gpu installed)
    vectors.npy      # OR raw numpy vectors (no faiss)
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

VECTOR_DIRNAME = "vector_store"
INDEX_FILENAME = "memory.index"
VECTORS_FILENAME = "vectors.npy"
RECORDS_FILENAME = "records.json"
MANIFEST_FILENAME = "manifest.json"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

_client_cache: dict[str, "EmbeddingClient"] = {}
_client_lock = threading.Lock()


class EmbeddingClient(Protocol):
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class MemoryRecord:
    id: str
    item_type: str
    text: str
    payload: dict[str, Any]


# ── embedding clients ────────────────────────────────────────────────────────

class SentenceTransformerEmbeddingClient:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"sentence-transformers is required for the default embedding "
                f"model ({model_name!r}). Install it with:\n"
                f"    pip install camroll-agent[embeddings]\n"
                f"or pick a different embedding model "
                f"(e.g. --embedding-model text-embedding-3-small to use OpenAI)."
            ) from exc
        self.model = SentenceTransformer(model_name, device="cpu")
        if any(p.is_meta for p in self.model.parameters()):
            self.model = self.model.to_empty(device="cpu")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True).tolist()


class OpenAIEmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                f"OpenAI embeddings ({model_name!r}) require the openai SDK. "
                f"Install it with:\n    pip install camroll-agent[openai]"
            ) from exc
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model_name = model_name

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in resp.data]


def _is_local_st(model_name: str) -> bool:
    """Heuristic: sentence-transformers IDs contain a slash and are not OpenAI."""
    openai_prefixes = ("text-embedding-", "ada-")
    return "/" in model_name and not any(model_name.startswith(p) for p in openai_prefixes)


def _build_client(model_name: str) -> EmbeddingClient:
    if _is_local_st(model_name):
        return SentenceTransformerEmbeddingClient(model_name)
    return OpenAIEmbeddingClient(model_name)


# ── build ────────────────────────────────────────────────────────────────────

def build_vector_index(
    output_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client: Optional[EmbeddingClient] = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    records = _load_records_from_outputs(output_path)
    if not records:
        raise ValueError(f"No event/image rows found in {output_path}")

    if embedding_client is None:
        embedding_client = _build_client(embedding_model)

    vectors = np.array(
        embedding_client.embed_many([r.text for r in records]),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError("Embedding client returned an unexpected vector shape.")
    _normalize_rows(vectors)

    backend = "faiss" if faiss is not None else "numpy"
    vector_dir = output_path / VECTOR_DIRNAME
    vector_dir.mkdir(parents=True, exist_ok=True)
    if backend == "faiss":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(vector_dir / INDEX_FILENAME))
    else:
        np.save(vector_dir / VECTORS_FILENAME, vectors)

    (vector_dir / RECORDS_FILENAME).write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
    )
    (vector_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "record_count": len(records),
                "dimension": int(vectors.shape[1]),
                "backend": backend,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return {
        "vector_dir": str(vector_dir),
        "record_count": len(records),
        "dimension": int(vectors.shape[1]),
        "embedding_model": embedding_model,
        "backend": backend,
    }


# ── query ────────────────────────────────────────────────────────────────────

def query_vector_index(
    output_dir: str | Path,
    query: str,
    *,
    top_k: int = 5,
    item_type: Optional[str] = None,
    embedding_client: Optional[EmbeddingClient] = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    vector_dir = output_path / VECTOR_DIRNAME
    manifest = json.loads((vector_dir / MANIFEST_FILENAME).read_text())
    records = [
        MemoryRecord(**item)
        for item in json.loads((vector_dir / RECORDS_FILENAME).read_text())
    ]

    candidate_count = min(max(top_k * 4, top_k), len(records))
    backend = manifest.get("backend")

    stored_model = manifest["embedding_model"]
    if embedding_client is None:
        with _client_lock:
            if stored_model not in _client_cache:
                _client_cache[stored_model] = _build_client(stored_model)
        embedding_client = _client_cache[stored_model]
    qv = np.array(embedding_client.embed_many([query]), dtype=np.float32)
    _normalize_rows(qv)

    if backend == "faiss":
        if faiss is None:
            raise ImportError("faiss is required to query this vector store backend.")
        index = faiss.read_index(str(vector_dir / INDEX_FILENAME))
        scores, indices = index.search(qv, candidate_count)
    else:
        vectors = np.load(vector_dir / VECTORS_FILENAME)
        scores_matrix = vectors @ qv[0]
        ranked = np.argsort(scores_matrix)[::-1][:candidate_count]
        scores = np.array([scores_matrix[ranked]], dtype=np.float32)
        indices = np.array([ranked], dtype=np.int64)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = records[int(idx)]
        if item_type and record.item_type != item_type:
            continue
        results.append({
            "score": float(score),
            "id": record.id,
            "item_type": record.item_type,
            "text": record.text,
            "payload": record.payload,
        })
        if len(results) >= top_k:
            break

    return {
        "query": query,
        "top_k": top_k,
        "item_type": item_type,
        "results": results,
        "vector_dir": str(vector_dir),
        "backend": backend,
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_records_from_outputs(output_dir: Path) -> list[MemoryRecord]:
    events_path = output_dir / "events.json"
    images_path = output_dir / "images.json"
    if not events_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"Expected events.json and images.json in {output_dir} "
            "(produced by camroll_agent.build_memory.run)."
        )
    events = json.loads(events_path.read_text())
    images = json.loads(images_path.read_text())
    return _build_event_records(events) + _build_image_records(images)


def _build_event_records(events: Iterable[dict[str, Any]]) -> list[MemoryRecord]:
    out: list[MemoryRecord] = []
    for idx, event in enumerate(events):
        payload = {
            "event": event["event"],
            "date": event.get("date"),
            "dates": event.get("dates", []),
            "description": event.get("description", ""),
            "images": event.get("images", []),
        }
        text = "\n".join([
            "Type: event",
            f"Event: {payload['event']}",
            f"Date: {payload['date']}",
            f"Dates: {', '.join(payload['dates']) if payload['dates'] else payload['date']}",
            f"Description: {payload['description']}",
        ])
        out.append(MemoryRecord(id=f"event::{idx}", item_type="event", text=text, payload=payload))
    return out


def _build_image_records(images: Iterable[dict[str, Any]]) -> list[MemoryRecord]:
    out: list[MemoryRecord] = []
    for idx, image in enumerate(images):
        payload = {
            "image": image.get("image"),
            "path": image["path"],
            "date": image.get("date"),
            "caption": image.get("caption", ""),
            "operation": image.get("operation"),
            "event": image.get("event"),
        }
        text = "\n".join([
            "Type: image",
            f"Image: {payload['image']}",
            f"Date: {payload['date']}",
            f"Event: {payload['event'] or 'none'}",
            f"Caption: {payload['caption']}",
        ])
        out.append(MemoryRecord(id=f"image::{idx}", item_type="image", text=text, payload=payload))
    return out


def _normalize_rows(vectors: np.ndarray) -> None:
    if faiss is not None:
        faiss.normalize_L2(vectors)
        return
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

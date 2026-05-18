"""Stage 2 — build the searchable index from events.json + images.json.

Produces, inside the same `memory_dir/`:
  metadata.sqlite    SQLite + FTS5 keyword index
  vector_store/      embeddings + FAISS / numpy / TF-IDF backend

Run after Stage 1 (camroll_agent.build_memory.run).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from camroll_agent import store as ms
from camroll_agent import vector as vec

DEFAULT_EMBEDDING_MODEL = vec.DEFAULT_EMBEDDING_MODEL


# ── FTS contextualization ────────────────────────────────────────────────────

def _event_fts_text(ev: dict) -> str:
    return f"{ev['event']}\n{ev.get('description', '')}"


def _image_fts_text(img: dict, event_name: str, event_desc: str) -> str:
    """Prepend event context to image caption so queries matching the event
    name surface the image even if the caption doesn't mention it directly
    (Anthropic-style contextualization)."""
    event_preview = event_desc[:200] if event_desc else ""
    parts = []
    if event_name:
        parts.append(f"[Event: {event_name}]")
    if event_preview:
        parts.append(event_preview)
    parts.append(img.get("caption", ""))
    return "\n".join(p for p in parts if p)


# ── public API ───────────────────────────────────────────────────────────────

def run(
    memory_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client: Any = None,
) -> dict[str, Any]:
    """Build metadata.sqlite + vector_store/ from events.json + images.json.

    Args:
        memory_dir: directory containing events.json and images.json
                    (Stage 1 output). Index files are written here.
        embedding_model: name of the embedding model (sentence-transformers
                         ID for local, or 'text-embedding-3-small' for OpenAI).
        embedding_client: an explicit EmbeddingClient. Overrides embedding_model.

    Returns: summary dict with counts + backend choice.
    """
    memory_dir = Path(memory_dir).expanduser().resolve()
    events = json.loads((memory_dir / "events.json").read_text())
    images = json.loads((memory_dir / "images.json").read_text())

    # ── SQLite + FTS5 ────────────────────────────────────────────────────
    event_by_name: dict[str, dict] = {}
    for ev in events:
        name = ev["event"]
        dates = ev.get("dates") or ([ev["date"]] if ev.get("date") else [])
        dates = sorted(d for d in dates if d and d != "unknown")
        date_start = dates[0] if dates else ev.get("date")
        date_end = dates[-1] if dates else ev.get("date")
        event_by_name[name] = {
            "event_id": ms.event_id_for(name, date_start or ""),
            "name": name,
            "date_start": date_start,
            "date_end": date_end,
            "description": ev.get("description", "") or "",
        }

    conn = ms.connect(memory_dir)
    try:
        ms.clear_all(conn)
        for name, ev in event_by_name.items():
            ms.insert_event(
                conn,
                event_id=ev["event_id"], name=ev["name"],
                date_start=ev["date_start"], date_end=ev["date_end"],
                description=ev["description"],
            )
            ms.insert_event_fts(conn, ev["event_id"], _event_fts_text({"event": name, "description": ev["description"]}))

        for img in images:
            path = img.get("path") or ""
            if not path:
                continue
            iid = ms.image_id_for(path)
            event_name = img.get("event") or ""
            ev_row = event_by_name.get(event_name)
            event_id = ev_row["event_id"] if ev_row else None
            event_desc = ev_row["description"] if ev_row else ""
            ms.insert_image(
                conn,
                image_id=iid, path=path,
                date=img.get("date"),
                event_id=event_id,
                caption=img.get("caption", "") or "",
            )
            ms.insert_image_fts(
                conn, iid,
                _image_fts_text(img, event_name, event_desc),
            )

        conn.commit()
        sqlite_stats = ms.stats(conn)
    finally:
        conn.close()

    # ── vector store ─────────────────────────────────────────────────────
    vec_summary = vec.build_vector_index(
        memory_dir,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
    )

    return {
        "memory_dir": str(memory_dir),
        "sqlite": sqlite_stats,
        "vector": vec_summary,
    }

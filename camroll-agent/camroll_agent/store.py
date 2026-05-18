"""SQLite + FTS5 metadata store.

Schema:
  events(event_id PK, name, date_start, date_end, description, location, people)
  images(image_id PK, path, date, time, location, people, event_id FK, caption)
  images_fts  FTS5 over (caption + event name/desc), contextualized
  events_fts  FTS5 over (name + description)

IDs:
  event_id  = "ev_<hash>"   (8 hex chars of sha1(event_name + date_start))
  image_id  = "img_<hash>"  (8 hex chars of sha1(path))
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

DB_FILENAME = "metadata.sqlite"


# ── id helpers ───────────────────────────────────────────────────────────────

def event_id_for(name: str, date_start: str) -> str:
    h = hashlib.sha1(f"{name}|{date_start}".encode("utf-8")).hexdigest()[:8]
    return f"ev_{h}"


def image_id_for(path: str) -> str:
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    return f"img_{h}"


def is_event_id(s: str) -> bool:
    return s.startswith("ev_")


def is_image_id(s: str) -> bool:
    return s.startswith("img_")


# ── schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    date_start   TEXT,
    date_end     TEXT,
    description  TEXT,
    location     TEXT,
    people       TEXT
);

CREATE TABLE IF NOT EXISTS images (
    image_id   TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    date       TEXT,
    time       TEXT,
    location   TEXT,
    people     TEXT,
    event_id   TEXT,
    caption    TEXT,
    FOREIGN KEY(event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_images_date     ON images(date);
CREATE INDEX IF NOT EXISTS idx_images_event_id ON images(event_id);
CREATE INDEX IF NOT EXISTS idx_events_date     ON events(date_start);

CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    image_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    event_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def connect(memory_dir: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(memory_dir) / DB_FILENAME
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found. Run camroll_agent.index.run(memory_dir) first."
            )
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def db_path(memory_dir: str | Path) -> Path:
    return Path(memory_dir) / DB_FILENAME


# ── write helpers ────────────────────────────────────────────────────────────

def insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    name: str,
    date_start: str | None,
    date_end: str | None,
    description: str,
    location: str = "",
    people: list[str] | None = None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO events
           (event_id, name, date_start, date_end, description, location, people)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (event_id, name, date_start, date_end, description, location,
         json.dumps(people or [], ensure_ascii=False)),
    )


def insert_image(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    path: str,
    date: str | None,
    time: str = "",
    location: str = "",
    people: list[str] | None = None,
    event_id: str | None,
    caption: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO images
           (image_id, path, date, time, location, people, event_id, caption)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (image_id, path, date, time, location,
         json.dumps(people or [], ensure_ascii=False), event_id, caption),
    )


def insert_image_fts(conn: sqlite3.Connection, image_id: str, text: str) -> None:
    conn.execute("INSERT INTO images_fts (image_id, text) VALUES (?, ?)",
                 (image_id, text))


def insert_event_fts(conn: sqlite3.Connection, event_id: str, text: str) -> None:
    conn.execute("INSERT INTO events_fts (event_id, text) VALUES (?, ?)",
                 (event_id, text))


def clear_all(conn: sqlite3.Connection) -> None:
    for t in ("images_fts", "events_fts", "images", "events"):
        conn.execute(f"DELETE FROM {t}")


# ── read helpers ─────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    if "people" in d and isinstance(d["people"], str):
        try:
            d["people"] = json.loads(d["people"]) if d["people"] else []
        except json.JSONDecodeError:
            d["people"] = []
    return d


def get_event(conn: sqlite3.Connection, event_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    event = _row_to_dict(row)
    if event is None:
        return None
    img_rows = conn.execute(
        "SELECT image_id FROM images WHERE event_id = ? ORDER BY date, image_id",
        (event_id,),
    ).fetchall()
    event["image_ids"] = [r["image_id"] for r in img_rows]
    return event


def get_image(conn: sqlite3.Connection, image_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()
    img = _row_to_dict(row)
    if img is None:
        return None
    if img.get("event_id"):
        ev = conn.execute(
            "SELECT name FROM events WHERE event_id = ?", (img["event_id"],),
        ).fetchone()
        img["event_name"] = ev["name"] if ev else None
    else:
        img["event_name"] = None
    return img


def image_context(
    conn: sqlite3.Connection, image_id: str, *, caption_preview_chars: int = 500,
) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT i.image_id, i.date, i.time, i.location, i.people,
                  i.event_id, e.name AS event_name, i.caption
             FROM images i
        LEFT JOIN events e ON i.event_id = e.event_id
            WHERE i.image_id = ?""",
        (image_id,),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    try:
        d["people"] = json.loads(d["people"]) if d.get("people") else []
    except json.JSONDecodeError:
        d["people"] = []
    caption = d.pop("caption", "") or ""
    d["caption_preview"] = (
        caption[:caption_preview_chars] + "…"
        if len(caption) > caption_preview_chars else caption
    )
    return d


def event_context(
    conn: sqlite3.Connection, event_id: str, *, desc_preview_chars: int = 500,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT event_id, name, date_start, date_end, location, people, description "
        "FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    try:
        d["people"] = json.loads(d["people"]) if d.get("people") else []
    except json.JSONDecodeError:
        d["people"] = []
    desc = d.pop("description", "") or ""
    d["description_preview"] = (
        desc[:desc_preview_chars] + "…"
        if len(desc) > desc_preview_chars else desc
    )
    return d


def list_images_by_date(
    conn: sqlite3.Connection,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    location: str | None = None,
    person: str | None = None,
    event_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)
    if location:
        clauses.append("location LIKE ?")
        params.append(f"%{location}%")
    if person:
        clauses.append("people LIKE ?")
        params.append(f"%{person}%")
    if event_id:
        clauses.append("event_id = ?")
        params.append(event_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT image_id FROM images {where} ORDER BY date, image_id LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [image_context(conn, r["image_id"]) for r in rows]


def list_events_by_date(
    conn: sqlite3.Connection,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    location: str | None = None,
    person: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if date_to:
        clauses.append("(date_start IS NULL OR date_start <= ?)")
        params.append(date_to)
    if date_from:
        clauses.append("(date_end   IS NULL OR date_end   >= ?)")
        params.append(date_from)
    if location:
        clauses.append("location LIKE ?")
        params.append(f"%{location}%")
    if person:
        clauses.append("people LIKE ?")
        params.append(f"%{person}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT event_id FROM events {where} ORDER BY date_start, event_id LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [event_context(conn, r["event_id"]) for r in rows]


def fts_search_images(
    conn: sqlite3.Connection, query: str, *, top_k: int = 10,
) -> list[dict[str, Any]]:
    fts_query = _escape_fts_query(query)
    rows = conn.execute(
        """SELECT image_id, bm25(images_fts) AS bm25_score,
                  snippet(images_fts, 1, '«', '»', '…', 12) AS snippet
             FROM images_fts
            WHERE images_fts MATCH ?
         ORDER BY bm25_score
            LIMIT ?""",
        (fts_query, top_k),
    ).fetchall()
    out = []
    for r in rows:
        ctx = image_context(conn, r["image_id"]) or {"image_id": r["image_id"]}
        ctx["score"] = round(-float(r["bm25_score"]), 4)
        ctx["snippet"] = r["snippet"]
        out.append(ctx)
    return out


def fts_search_events(
    conn: sqlite3.Connection, query: str, *, top_k: int = 10,
) -> list[dict[str, Any]]:
    fts_query = _escape_fts_query(query)
    rows = conn.execute(
        """SELECT event_id, bm25(events_fts) AS bm25_score,
                  snippet(events_fts, 1, '«', '»', '…', 12) AS snippet
             FROM events_fts
            WHERE events_fts MATCH ?
         ORDER BY bm25_score
            LIMIT ?""",
        (fts_query, top_k),
    ).fetchall()
    out = []
    for r in rows:
        ctx = event_context(conn, r["event_id"]) or {"event_id": r["event_id"]}
        ctx["score"] = round(-float(r["bm25_score"]), 4)
        ctx["snippet"] = r["snippet"]
        out.append(ctx)
    return out


def fts_count_images(conn: sqlite3.Connection, query: str) -> int:
    fts_query = _escape_fts_query(query)
    return int(conn.execute(
        "SELECT COUNT(*) AS c FROM images_fts WHERE images_fts MATCH ?",
        (fts_query,),
    ).fetchone()["c"])


def fts_count_events(conn: sqlite3.Connection, query: str) -> int:
    fts_query = _escape_fts_query(query)
    return int(conn.execute(
        "SELECT COUNT(*) AS c FROM events_fts WHERE events_fts MATCH ?",
        (fts_query,),
    ).fetchone()["c"])


def _escape_fts_query(query: str) -> str:
    tokens = []
    for tok in query.split():
        cleaned = tok.replace('"', '').strip()
        if cleaned:
            tokens.append(f'"{cleaned}"')
    return " ".join(tokens) if tokens else '"__no_match_placeholder__"'


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    n_events = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
    n_images = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()["c"]
    row = conn.execute(
        "SELECT MIN(date) AS a, MAX(date) AS b FROM images",
    ).fetchone()
    return {
        "n_events": n_events,
        "n_images": n_images,
        "date_earliest": row["a"],
        "date_latest": row["b"],
    }

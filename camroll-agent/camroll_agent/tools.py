"""The 5 atomic tools that the camroll agent calls.

Pure functions. No LLM, no agent loop. All return JSON-serializable dicts.
Testable in isolation:

    from camroll_agent.tools import search
    out = search(thought="…", query="lake michigan", memory_dir="memory/")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from camroll_agent import store as ms
from camroll_agent import vector as vec

MAX_IMAGES_PER_CALL = 6


# ── 1. search ─────────────────────────────────────────────────────────

def search(
    *,
    thought: str,
    query: str,
    memory_dir: str,
    top_k: int = 10,
    kind: str = "both",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Semantic (vector) search over events and/or image captions."""
    del thought
    kind = kind.lower().strip()
    assert kind in ("both", "events", "captions", "images"), f"bad kind={kind}"

    if date_from or date_to:
        pool_k = max(top_k * 20, 200)
    elif kind != "both":
        pool_k = top_k * 4
    else:
        pool_k = top_k

    raw = vec.query_vector_index(memory_dir, query=query, top_k=pool_k, item_type=None)
    conn = ms.connect(memory_dir, read_only=True)
    try:
        out: list[dict[str, Any]] = []
        for r in raw["results"]:
            item_type = r["item_type"]
            if kind == "events" and item_type != "event":
                continue
            if kind in ("captions", "images") and item_type != "image":
                continue

            payload = r["payload"]
            if item_type == "image":
                path = payload.get("path", "")
                iid = ms.image_id_for(path) if path else None
                ctx = ms.image_context(conn, iid) if iid else None
                if ctx is None:
                    continue
                if date_from and (ctx.get("date") or "") < date_from:
                    continue
                if date_to and (ctx.get("date") or "9999") > date_to:
                    continue
                ctx["kind"] = "image"
                ctx["id"] = ctx["image_id"]
                ctx["score"] = round(float(r["score"]), 4)
                out.append(ctx)
            else:
                name = payload.get("event") or payload.get("name") or ""
                row = conn.execute(
                    "SELECT event_id FROM events WHERE name = ? LIMIT 1", (name,),
                ).fetchone()
                if row is None:
                    continue
                ctx = ms.event_context(conn, row["event_id"])
                if ctx is None:
                    continue
                ds, de = ctx.get("date_start"), ctx.get("date_end")
                if date_from and (de or "9999") < date_from:
                    continue
                if date_to and (ds or "") > date_to:
                    continue
                ctx["kind"] = "event"
                ctx["id"] = ctx["event_id"]
                ctx["score"] = round(float(r["score"]), 4)
                out.append(ctx)

            if len(out) >= top_k:
                break
    finally:
        conn.close()

    return {
        "query": query, "kind": kind, "top_k": top_k,
        "date_from": date_from, "date_to": date_to,
        "count": len(out), "results": out,
    }


# ── 2. grep (BM25 via FTS5) ──────────────────────────────────────────────────

def grep(
    *,
    thought: str,
    query: str,
    memory_dir: str,
    kind: str = "both",
    top_k: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    del thought
    kind = kind.lower().strip()
    assert kind in ("both", "events", "captions", "images"), f"bad kind={kind}"

    pool_k = max(top_k * 20, 200) if (date_from or date_to) else top_k

    def _in_range_image(r: dict) -> bool:
        d = r.get("date") or ""
        if date_from and d < date_from:
            return False
        if date_to and d and d > date_to:
            return False
        return True

    def _in_range_event(r: dict) -> bool:
        ds = r.get("date_start") or ""
        de = r.get("date_end") or ds
        if date_from and (de or "9999") < date_from:
            return False
        if date_to and (ds or "") > date_to:
            return False
        return True

    conn = ms.connect(memory_dir, read_only=True)
    try:
        events_count = images_count = 0
        results: list[dict[str, Any]] = []
        if kind in ("both", "events"):
            events_count = ms.fts_count_events(conn, query)
            for r in ms.fts_search_events(conn, query, top_k=pool_k):
                if (date_from or date_to) and not _in_range_event(r):
                    continue
                r["kind"] = "event"
                r["id"] = r["event_id"]
                results.append(r)
        if kind in ("both", "captions", "images"):
            images_count = ms.fts_count_images(conn, query)
            for r in ms.fts_search_images(conn, query, top_k=pool_k):
                if (date_from or date_to) and not _in_range_image(r):
                    continue
                r["kind"] = "image"
                r["id"] = r["image_id"]
                results.append(r)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:top_k]
    finally:
        conn.close()

    return {
        "query": query, "kind": kind,
        "date_from": date_from, "date_to": date_to,
        "count_events": events_count, "count_images": images_count,
        "count_total": events_count + images_count,
        "results": results,
    }


# ── 3. list_by_date ──────────────────────────────────────────────────────────

def list_by_date(
    *,
    thought: str,
    memory_dir: str,
    date_from: str | None = None,
    date_to: str | None = None,
    kind: str = "both",
    location: str | None = None,
    person: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    del thought
    kind = kind.lower().strip()
    assert kind in ("both", "events", "captions", "images"), f"bad kind={kind}"

    conn = ms.connect(memory_dir, read_only=True)
    try:
        events: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        if kind in ("both", "events"):
            events = ms.list_events_by_date(
                conn, date_from=date_from, date_to=date_to,
                location=location, person=person, limit=limit, offset=offset,
            )
            for e in events:
                e["kind"] = "event"
                e["id"] = e["event_id"]
        if kind in ("both", "captions", "images"):
            images = ms.list_images_by_date(
                conn, date_from=date_from, date_to=date_to,
                location=location, person=person, limit=limit, offset=offset,
            )
            for i in images:
                i["kind"] = "image"
                i["id"] = i["image_id"]
    finally:
        conn.close()

    return {
        "date_from": date_from, "date_to": date_to,
        "kind": kind, "location": location, "person": person,
        "count_events": len(events), "count_images": len(images),
        "results": events + images,
    }


# ── 4. get ───────────────────────────────────────────────────────────────────

def get(*, thought: str, id: str, memory_dir: str) -> dict[str, Any]:
    del thought
    conn = ms.connect(memory_dir, read_only=True)
    try:
        if ms.is_event_id(id):
            rec = ms.get_event(conn, id)
            if rec is None:
                return {"error": f"event_id not found: {id}"}
            rec["kind"] = "event"
            rec["id"] = rec["event_id"]
            return rec
        if ms.is_image_id(id):
            rec = ms.get_image(conn, id)
            if rec is None:
                return {"error": f"image_id not found: {id}"}
            rec["kind"] = "image"
            rec["id"] = rec["image_id"]
            return rec
        return {"error": f"unrecognized id format (expected 'ev_…' or 'img_…'): {id}"}
    finally:
        conn.close()


# ── 5. view_image ────────────────────────────────────────────────────────────

def view_image(
    *,
    thought: str,
    image_ids: list[str] | str,
    prompt: str,
    memory_dir: str,
    image_client,
) -> dict[str, Any]:
    """Look at one or more images with an agent-authored prompt.

    Counts as ONE VLM call regardless of list size, so the agent can batch
    comparisons. The list is capped at MAX_IMAGES_PER_CALL.
    """
    del thought
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    if not image_ids:
        return {"error": "image_ids cannot be empty"}
    if len(image_ids) > MAX_IMAGES_PER_CALL:
        image_ids = image_ids[:MAX_IMAGES_PER_CALL]

    conn = ms.connect(memory_dir, read_only=True)
    try:
        contexts, paths, skipped = [], [], []
        for iid in image_ids:
            if not ms.is_image_id(iid):
                skipped.append({"image_id": iid, "reason": "not an image id"})
                continue
            img = ms.get_image(conn, iid)
            if img is None:
                skipped.append({"image_id": iid, "reason": "not found"})
                continue
            path = img.get("path", "")
            if not path or not Path(path).exists():
                skipped.append({
                    "image_id": iid, "reason": "file not on disk", "path": path,
                })
                continue
            ctx = ms.image_context(conn, iid) or {}
            ctx["kind"] = "image"
            ctx["id"] = iid
            contexts.append(ctx)
            paths.append(path)
    finally:
        conn.close()

    if not paths:
        return {"error": "no viewable images", "skipped": skipped}

    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        bits = [f"image {i}", f"id={c['id']}"]
        if c.get("date"):
            bits.append(f"date={c['date']}")
        if c.get("event_name"):
            bits.append(f"event={c['event_name']}")
        if c.get("location"):
            bits.append(f"location={c['location']}")
        ctx_lines.append(" | ".join(bits))

    full_prompt = (
        f"You are looking at {len(paths)} photo(s). Per-photo context:\n"
        + "\n".join(ctx_lines)
        + f"\n\nThe user's question about these photo(s):\n\"{prompt}\"\n\n"
        f"Answer factually and concretely. If different photos show different "
        f"things, say which photo (by its id) shows what."
    )

    if image_client is None:
        return {
            "error": "view_image disabled (no image_client configured)",
            "images": contexts,
            "skipped": skipped,
        }

    try:
        analysis = image_client.generate(full_prompt, paths)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"VLM call failed: {exc}",
            "images": contexts,
            "skipped": skipped,
        }

    return {"analysis": analysis, "images": contexts, "skipped": skipped}

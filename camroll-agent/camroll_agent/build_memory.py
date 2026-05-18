"""Stage 1 — build a structured memory from a conversation JSON.

For each photo in the conversation (in chronological order), we ask a VLM to:
  - Write a first-person caption for the photo.
  - Decide whether the photo belongs to a new event (ADD), the current
    event (UPDATE), or doesn't change the event table (NO_OP).

The output is written to `output_dir/`:
  events.json      list of events (name, dates, description, image paths)
  images.json      list of images (path, date, caption, event)
  operations.json  one record per VLM call (for inspection / resume)

Input conversation JSON shape:

    {
      "root_folder": "/abs/path/to/photos",     # optional
      "profile_image": "profile.jpg",           # absolute or relative to root_folder
      "library_description": "A 2005-2013 album from a college student.",
      "turns": [
        {"date": "2005-10-01", "user": {"image": "847410131.jpg"}},
        {"date": "2005-10-01", "user": {"image": "847410831.jpg"}},
        ...
      ]
    }

Each turn's `user.image` is the photo to caption. `date` is the photo's date.
You can include extra metadata fields on the turn; they'll be passed through
to the VLM prompt as additional context.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from camroll_agent.llm import VLMClient, build_vlm

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
OPERATIONS = {"ADD", "UPDATE", "NO_OP"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class EventRow:
    event: str
    date: str
    description: str
    dates: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


@dataclass
class ImageRow:
    image: str
    caption: str
    path: str
    date: str
    operation: str
    event: Optional[str] = None


@dataclass
class StepRecord:
    step: int
    total: int
    image: str
    path: str
    date: str
    operation: str
    event: Optional[str]
    parsed_output: dict[str, Any]
    raw_response: str


# ── conversation loading ─────────────────────────────────────────────────────

def _resolve_image_path(
    raw: str, root_folder: str | None, pattern: str | None,
) -> Path:
    """Resolve a (possibly relative) image path against the JSON's root_folder."""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if root_folder:
        tpl = pattern or "{root_folder}/{relative_path}"
        joined = tpl.format(
            root_folder=str(root_folder).rstrip("/"),
            relative_path=raw,
        )
        return Path(joined).expanduser().resolve()
    return candidate.resolve()


def _load_conversation(spec_path: Path) -> dict[str, Any]:
    data = json.loads(spec_path.read_text())
    root_folder = data.get("root_folder")
    image_path_pattern = data.get("image_path_pattern")

    profile_raw = (
        data.get("profile_image")
        or data.get("profile_image_path")
        or data.get("profile_photo")
    )
    if not profile_raw:
        raise ValueError(
            "Conversation JSON must include `profile_image` "
            "(absolute path or relative to `root_folder`)."
        )
    profile_path = _resolve_image_path(profile_raw, root_folder, image_path_pattern)
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile_image not found: {profile_path}")

    image_paths: list[Path] = []
    date_lookup: dict[str, str] = {}
    extra_lookup: dict[str, dict[str, Any]] = {}
    for turn in data.get("turns", []):
        user = turn.get("user") or {}
        raw_image = user.get("image")
        if not raw_image:
            continue
        p = _resolve_image_path(raw_image, root_folder, image_path_pattern)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        image_paths.append(p)
        key = str(p)
        date_lookup[key] = str(turn.get("date", "")).strip() or "unknown"
        extras = {
            k: v for k, v in turn.items()
            if k not in ("user", "date") and v not in (None, "", [], {})
        }
        if extras:
            extra_lookup[key] = extras

    return {
        "profile_image_path": profile_path,
        "library_description": (
            data.get("library_description")
            or data.get("photo_library_description")
            or data.get("user_description")
            or data.get("description")
            or None
        ),
        "image_paths": image_paths,
        "date_lookup": date_lookup,
        "extra_lookup": extra_lookup,
        "lookback_k": int(data.get("lookback_k", 3)),
    }


# ── public API ───────────────────────────────────────────────────────────────

def inspect(spec_path: str | Path) -> dict[str, Any]:
    """Print info about a conversation JSON without running the VLM."""
    spec_path = Path(spec_path).expanduser().resolve()
    conv = _load_conversation(spec_path)
    image_paths = conv["image_paths"]
    dated = [
        conv["date_lookup"][str(p)] for p in image_paths
        if conv["date_lookup"].get(str(p)) and conv["date_lookup"][str(p)] != "unknown"
    ]
    unique = sorted(set(dated))
    return {
        "conversation_json": str(spec_path),
        "profile_image": str(conv["profile_image_path"]),
        "library_description": conv["library_description"],
        "image_count": len(image_paths),
        "dated_image_count": len(dated),
        "unique_date_count": len(unique),
        "date_range": (
            "unknown" if not unique
            else unique[0] if len(unique) == 1
            else f"{unique[0]} to {unique[-1]}"
        ),
        "first_image": str(image_paths[0]) if image_paths else None,
        "last_image": str(image_paths[-1]) if image_paths else None,
    }


def run(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    vlm: VLMClient | None = None,
    backend: str = "openai",
    model: str | None = None,
    max_images: int | None = None,
    resume: bool = False,
    checkpoint_every: int = 50,
) -> dict[str, Any]:
    """Run Stage 1: caption every photo + group them into events.

    Args:
        spec_path: path to the conversation JSON.
        output_dir: directory where events.json / images.json / operations.json
                    will be written. Created if missing.
        vlm: an explicit VLMClient (e.g. a custom subclass). If None, a client
             is built from `backend` + `model`.
        backend: which built-in VLM to use ("openai", "gemini", or "local").
                 Ignored if `vlm` is set.
        model: model name for the chosen backend.
        max_images: only process the first N images (useful for smoke tests).
        resume: continue an interrupted run in `output_dir`.
        checkpoint_every: flush intermediate JSON every N images.

    Returns: a summary dict with output paths and counts.
    """
    spec_path = Path(spec_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conv = _load_conversation(spec_path)
    image_paths = conv["image_paths"]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No images found in {spec_path}")
    date_lookup = {
        str(p): conv["date_lookup"].get(str(p), "unknown") for p in image_paths
    }

    if resume:
        event_rows, image_rows, operation_log = _load_existing(output_dir)
        _validate_resume(image_paths, image_rows, output_dir)
    else:
        event_rows, image_rows, operation_log = [], [], []

    _backfill_event_dates(event_rows, date_lookup)

    if vlm is None:
        vlm = build_vlm(backend, model)

    start_index = len(image_rows)
    remaining = image_paths[start_index:]
    total = len(image_paths)
    profile_path = str(conv["profile_image_path"])
    lookback_k = conv["lookback_k"]
    library_description = conv["library_description"]

    bar = tqdm(remaining, total=len(remaining), desc="captioning")
    for offset, image_path in enumerate(bar, start=1):
        index = start_index + offset
        current_date = date_lookup.get(str(image_path), "unknown")
        extra = conv["extra_lookup"].get(str(image_path))
        try:
            decision = _run_step(
                vlm=vlm,
                profile_path=profile_path,
                current_image=image_path,
                current_date=current_date,
                lookback_k=lookback_k,
                library_description=library_description,
                image_rows=image_rows,
                event_rows=event_rows,
                extras=extra,
            )
        except Exception as exc:
            print(f"\n[build_memory] WARNING: skipping {image_path.name} ({type(exc).__name__}: {exc})")
            decision = {
                "operation": "NO_OP",
                "image_caption": f"[skipped: {type(exc).__name__}]",
                "event": None,
                "parsed_output": {},
                "raw_response": str(exc),
            }

        row = ImageRow(
            image=image_path.name,
            caption=decision["image_caption"],
            path=str(image_path),
            date=current_date,
            operation=decision["operation"],
        )
        event_name = _apply_event_operation(
            decision=decision,
            event_rows=event_rows,
            current_image=image_path,
            date_lookup=date_lookup,
        )
        if event_name:
            row.event = event_name
        image_rows.append(row)
        operation_log.append(StepRecord(
            step=index, total=total,
            image=row.image, path=row.path, date=row.date,
            operation=row.operation, event=row.event,
            parsed_output=decision["parsed_output"],
            raw_response=decision["raw_response"],
        ))

        if index % checkpoint_every == 0:
            _write_outputs(output_dir, event_rows, image_rows, operation_log)

    _write_outputs(output_dir, event_rows, image_rows, operation_log)

    return {
        "output_dir": str(output_dir),
        "n_events": len(event_rows),
        "n_images": len(image_rows),
        "events_json": str(output_dir / "events.json"),
        "images_json": str(output_dir / "images.json"),
    }


# ── per-image step ───────────────────────────────────────────────────────────

def _run_step(
    *,
    vlm: VLMClient,
    profile_path: str,
    current_image: Path,
    current_date: str,
    lookback_k: int,
    library_description: str | None,
    image_rows: list[ImageRow],
    event_rows: list[EventRow],
    extras: dict[str, Any] | None,
) -> dict[str, Any]:
    recent = image_rows[-lookback_k:] if lookback_k else []
    latest_event = event_rows[-1] if event_rows else None
    prompt = _build_prompt(
        current_date=current_date,
        lookback_k=lookback_k,
        library_description=library_description,
        recent_rows=recent,
        latest_event=latest_event,
        extras=extras,
    )
    response = vlm.generate(prompt, [profile_path, str(current_image)])
    payload = _parse_json_response(response)

    operation = str(payload.get("operation", "NO_OP")).upper().strip()
    if operation not in OPERATIONS:
        operation = "NO_OP"

    image_caption = str(payload.get("image_caption", "")).strip()
    if not image_caption:
        raise ValueError(f"Empty image_caption in VLM response for {current_image}")

    event_payload = payload.get("event")
    if not isinstance(event_payload, dict):
        event_payload = None
    elif "event_name" in event_payload and "event" not in event_payload:
        event_payload = {**event_payload, "event": event_payload["event_name"]}

    return {
        "operation": operation,
        "image_caption": image_caption,
        "event": event_payload,
        "parsed_output": payload,
        "raw_response": response,
    }


def _build_prompt(
    *,
    current_date: str,
    lookback_k: int,
    library_description: str | None,
    recent_rows: list[ImageRow],
    latest_event: Optional[EventRow],
    extras: dict[str, Any] | None,
) -> str:
    recent_payload = [asdict(row) for row in recent_rows]
    latest_event_payload = (
        {"event": latest_event.event, "description": latest_event.description}
        if latest_event else None
    )
    description = library_description or "No extra album description was provided."
    extras_block = json.dumps(extras or {}, indent=2, ensure_ascii=False)

    return f"""You are maintaining long-term structured memory for one user's personal photo library.

You will see two images:
1. The user's profile photo.
2. The current image that must be processed.

Important:
- Use the first image only as identity / reference. It tells you what the user looks like.
- Write the caption and event decision from the perspective of the user in the first image.
- The second image is the only source for `image_caption`, `operation`, and event reasoning.
- Never describe the first image in `image_caption`.

The album is processed in chronological order from oldest to newest.
Only update the most recent event row if the current image clearly belongs to it.

Event definition:
- An event is an episodic memory unit: a trip, outing, meal, hangout, celebration, class activity, etc.
- An event can span multiple consecutive photos with different dates, locations, subjects, or close-ups, as long as they still belong to the same broader episode (e.g., a road trip, a hangout with friends).
- Event names should summarize the broader episode, not the most eye-catching object in a single frame.

Album description:
{description}

Current image metadata:
- date: {current_date}
- extra: {extras_block}

Recent image table rows (up to the last {lookback_k}):
{json.dumps(recent_payload, indent=2, ensure_ascii=False)}

Latest event summary:
{json.dumps(latest_event_payload, indent=2, ensure_ascii=False)}

Tasks:
1. Write a detailed personalized caption for image 2 as if the person in image 1 is describing their own photo in first person.
2. Choose exactly one operation for the event table:
   - ADD: create a new event row
   - UPDATE: update the latest event row
   - NO_OP: do not modify the event table
3. If you choose ADD or UPDATE, return a full event row with fields:
   - event_name
   - description
   - date
   - images

Rules:
- The image caption must always be present.
- Use first-person wording when natural.
- The caption must describe image 2 only, not the reference image.
- If you don't know the person identity in image 2, use general wording ("a friend", "my mom", "a man", etc.) instead of inventing a name.
- Mention the important visible content in image 2: setting, people, objects, activity, atmosphere.
- Ground the caption in visible evidence; do not invent precise facts.
- For ADD or UPDATE, the event row's `images` must include the current image path.
- Prefer UPDATE when the current image is still part of the same broader episode.
- Prefer ADD only when there is a real episode boundary (change in outing, venue, social context, major activity, separate occasion).
- Event description should be at most 300 words; summarize / condense it over time.
- Return valid JSON only. No markdown fences.

JSON schema:
{{
  "operation": "ADD" | "UPDATE" | "NO_OP",
  "image_caption": "string",
  "event": {{
    "event_name": "string",
    "description": "string",
    "date": "string",
    "images": ["full image path", "..."]
  }} | null
}}
""".strip()


def _parse_json_response(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON: {response_text}")
        return json.loads(text[start: end + 1])


# ── event-table bookkeeping ──────────────────────────────────────────────────

def _apply_event_operation(
    *,
    decision: dict[str, Any],
    event_rows: list[EventRow],
    current_image: Path,
    date_lookup: dict[str, str],
) -> Optional[str]:
    operation = decision["operation"]
    event_payload = decision["event"] or {}
    if operation == "NO_OP":
        return None

    event_name = str(
        event_payload.get("event_name") or event_payload.get("event", "")
    ).strip() or "Untitled event"
    description = (
        str(event_payload.get("description", "")).strip()
        or decision["image_caption"]
    )
    provided_images = event_payload.get("images", [])
    images = [str(it) for it in provided_images if str(it).strip()]
    if operation == "UPDATE" and event_rows:
        images = event_rows[-1].images + images
    images.append(str(current_image))
    images = list(dict.fromkeys(images))
    dates = _collect_known_dates(images, date_lookup)
    date_value = _summarize_dates(dates)
    if date_value == "unknown":
        date_value = str(event_payload.get("date", "")).strip() or "unknown"
    if not dates and date_value != "unknown":
        dates = sorted({v for v in DATE_PATTERN.findall(date_value) if v != "unknown"})

    row = EventRow(
        event=event_name, date=date_value, dates=dates,
        description=description, images=images,
    )
    if operation == "ADD" or not event_rows:
        event_rows.append(row)
        return row.event

    # UPDATE: preserve existing name / description if not provided.
    if not (event_payload.get("event_name") or event_payload.get("event")):
        row.event = event_rows[-1].event
    if not event_payload.get("description"):
        row.description = event_rows[-1].description
    event_rows[-1] = row
    return row.event


def _collect_known_dates(image_paths: list[str], date_lookup: dict[str, str]) -> list[str]:
    return sorted({
        date_lookup[p] for p in image_paths
        if date_lookup.get(p) and date_lookup[p] != "unknown"
    })


def _summarize_dates(dates: list[str]) -> str:
    if not dates:
        return "unknown"
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]} to {dates[-1]}"


def _backfill_event_dates(event_rows: list[EventRow], date_lookup: dict[str, str]) -> None:
    for row in event_rows:
        if row.images:
            row.dates = _collect_known_dates(row.images, date_lookup)
        if not row.dates and row.date != "unknown":
            extracted = DATE_PATTERN.findall(row.date)
            row.dates = sorted({v for v in extracted if v != "unknown"})
        row.date = _summarize_dates(row.dates) if row.dates else (row.date or "unknown")


# ── resume / persistence ─────────────────────────────────────────────────────

def _load_existing(output_dir: Path):
    events_path = output_dir / "events.json"
    images_path = output_dir / "images.json"
    ops_path = output_dir / "operations.json"
    if not events_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"Cannot resume from {output_dir}: missing events.json / images.json."
        )
    events_payload = json.loads(events_path.read_text())
    images_payload = json.loads(images_path.read_text())

    event_rows = [
        EventRow(
            event=str(item.get("event", "")).strip() or "Untitled event",
            date=str(item.get("date", "")).strip() or "unknown",
            dates=_normalize_dates(item.get("dates")),
            description=str(item.get("description", "")).strip(),
            images=[str(p) for p in item.get("images", []) if str(p).strip()],
        )
        for item in events_payload
    ]
    image_rows = [
        ImageRow(
            image=str(it.get("image", "")).strip(),
            caption=str(it.get("caption", "")).strip(),
            path=str(it.get("path", "")).strip(),
            date=str(it.get("date", "")).strip() or "unknown",
            operation=str(it.get("operation", "NO_OP")).strip() or "NO_OP",
            event=str(it.get("event", "")).strip() or None,
        )
        for it in images_payload
    ]

    operation_log: list[StepRecord] = []
    if ops_path.exists():
        ops_payload = json.loads(ops_path.read_text())
        if isinstance(ops_payload, list) and len(ops_payload) == len(image_rows):
            for idx, (item, row) in enumerate(zip(ops_payload, image_rows), start=1):
                operation_log.append(StepRecord(
                    step=int(item.get("step", idx)),
                    total=int(item.get("total", 0)),
                    image=str(item.get("image", row.image)),
                    path=str(item.get("path", row.path)),
                    date=str(item.get("date", row.date)),
                    operation=str(item.get("operation", row.operation)),
                    event=str(item.get("event", "")) or row.event,
                    parsed_output=item.get("parsed_output", {}),
                    raw_response=str(item.get("raw_response", "")),
                ))
    if not operation_log:
        operation_log = [
            StepRecord(
                step=idx, total=0,
                image=row.image, path=row.path, date=row.date,
                operation=row.operation, event=row.event,
                parsed_output={}, raw_response="",
            )
            for idx, row in enumerate(image_rows, start=1)
        ]
    return event_rows, image_rows, operation_log


def _normalize_dates(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return sorted({
        str(v).strip() for v in raw
        if str(v).strip() and str(v).strip() != "unknown"
    })


def _validate_resume(image_paths, image_rows, output_dir: Path) -> None:
    if not image_rows:
        return
    if len(image_rows) > len(image_paths):
        raise ValueError(
            f"Cannot resume from {output_dir}: existing run has {len(image_rows)} "
            f"images but current selection only includes {len(image_paths)}."
        )
    expected = [str(p) for p in image_paths[:len(image_rows)]]
    actual = [row.path for row in image_rows]
    if actual != expected:
        raise ValueError(
            f"Cannot resume from {output_dir}: existing images.json does not "
            "match the current chronological prefix of selected images."
        )


def _write_outputs(
    output_dir: Path,
    event_rows: list[EventRow],
    image_rows: list[ImageRow],
    operation_log: list[StepRecord],
) -> None:
    (output_dir / "events.json").write_text(
        json.dumps([asdict(r) for r in event_rows], indent=2, ensure_ascii=False)
    )
    (output_dir / "images.json").write_text(
        json.dumps([asdict(r) for r in image_rows], indent=2, ensure_ascii=False)
    )
    (output_dir / "operations.json").write_text(
        json.dumps([asdict(r) for r in operation_log], indent=2, ensure_ascii=False)
    )
    _write_csv(
        output_dir / "events.csv",
        ["event", "date", "dates", "description", "images"],
        [
            {
                "event": r.event, "date": r.date,
                "dates": " | ".join(r.dates),
                "description": r.description,
                "images": " | ".join(r.images),
            }
            for r in event_rows
        ],
    )
    _write_csv(
        output_dir / "images.csv",
        ["image", "caption", "path", "date", "operation", "event"],
        [asdict(r) for r in image_rows],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

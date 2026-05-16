#!/usr/bin/env python3
"""
Build the public release tree for the camroll-yfcc20 dataset.

Source:   /sensei-fs-3/.../thaodata/yfcc_v3/{1..20}/
Output:   ./release/u{1..20}/

Per-user output layout:
    u{N}/
        images/             original Flickr JPEGs, copied as-is
        profile.jpg         user avatar
        qa.json             {"semantic_qa": [...], "episodic_qa": [...]}
                            image_ref values normalized to full "<id>.jpg" filenames
                            QA entries with ALL image_refs missing on disk are dropped
        photos.csv          Flickr metadata, English columns only

Top-level output:
    users.json              lightweight per-user manifest (id, nsid, nickname, years, counts)

Run:
    python3 prep_release.py                # all 20 users
    python3 prep_release.py --users 1      # just u1 (smoke test)
    python3 prep_release.py --users 1,2,3  # subset
    python3 prep_release.py --no-images    # skip image copy (fast metadata-only run)
"""

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_ROOT = Path("/sensei-fs-3/tenants/Sensei-AdobeResearchTeam/thaon/code/thaodata/yfcc_v3")
OUT_ROOT = Path(__file__).resolve().parent / "release"

ENGLISH_CSV_COLS = [
    "photoid", "uid", "unickname", "datetaken", "dateuploaded", "capturedevice",
    "title", "description", "usertags", "machinetags",
    "longitude", "latitude", "accuracy",
    "pageurl", "downloadurl", "licensename", "licenseurl",
    "serverid", "farmid", "secret", "secretoriginal", "ext", "marker",
]


def normalize_ref(ref: str, available_stems: dict[str, str]) -> str | None:
    """Map an image_ref (with or without .jpg) to the actual filename on disk."""
    if not ref:
        return None
    if ref in available_stems.values():
        return ref
    stem = os.path.splitext(ref)[0]
    return available_stems.get(stem)


def find_user_csv(info_dir: Path) -> Path:
    csvs = [p for p in info_dir.glob("user_*.csv")]
    if len(csvs) != 1:
        raise RuntimeError(f"expected exactly 1 user_*.csv in {info_dir}, got {csvs}")
    return csvs[0]


def parse_year_range(csv_path: Path) -> tuple[str | None, int]:
    """Return (years_str, row_count) from datetaken column."""
    years = []
    n = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            dt = row.get("datetaken", "")
            if dt and len(dt) >= 4 and dt[:4].isdigit():
                years.append(int(dt[:4]))
    if not years:
        return None, n
    lo, hi = min(years), max(years)
    return (f"{lo}" if lo == hi else f"{lo} – {hi}"), n


def _decode(s: str | None) -> str | None:
    """The Flickr CSV is URL-encoded (spaces as +, special chars as %XX). Decode for human-readable output."""
    if s is None:
        return None
    return urllib.parse.unquote_plus(s)


def write_english_csv(src_csv: Path, dst_csv: Path) -> tuple[str | None, str | None]:
    """Copy CSV with only the English columns (URL-decoded); return (nsid, nickname)."""
    nsid = None
    nickname = None
    text_cols = {"title", "description", "usertags", "machinetags", "unickname", "capturedevice"}
    with src_csv.open() as fin, dst_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=ENGLISH_CSV_COLS)
        writer.writeheader()
        for row in reader:
            decoded = {k: (_decode(row.get(k, "")) if k in text_cols else row.get(k, ""))
                       for k in ENGLISH_CSV_COLS}
            if nsid is None:
                nsid = decoded.get("uid")
                nickname = decoded.get("unickname")
            writer.writerow(decoded)
    return nsid, nickname


def build_qa(album_path: Path, image_dir: Path) -> tuple[dict, dict]:
    """
    Read album_data.json and produce flattened QA + stats.

    Returns (qa_dict, stats_dict).
    qa_dict = {"semantic_qa": [...], "episodic_qa": [...]}
    image_ref values are normalized to full filenames; entries whose
    image_refs ALL fail to resolve are dropped.
    """
    with album_path.open() as f:
        raw = json.load(f)

    available = {p.name for p in image_dir.iterdir() if p.is_file()}
    stems = {os.path.splitext(n)[0]: n for n in available}

    semantic = raw.get("semantic_qa", []) or []
    semantic_clean = []
    for q in semantic:
        semantic_clean.append({
            "question": q.get("question"),
            "answer": q.get("answer"),
            "detail_type": bool(q.get("detail_type", False)),
        })

    episodic_raw = raw.get("episodic_qa", []) or []
    episodic_clean = []
    qas_no_evidence = 0       # source had empty image_ref (question answerable from roll as a whole)
    qas_evidence_lost = 0     # source had refs, but no ref resolved to a file on disk
    refs_dropped = 0          # individual refs that didn't resolve (incl. partial)
    for q in episodic_raw:
        refs_in = q.get("image_ref", []) or []
        refs_out = []
        for r in refs_in:
            norm = normalize_ref(r, stems)
            if norm is None:
                refs_dropped += 1
                continue
            refs_out.append(norm)
        if not refs_in:
            qas_no_evidence += 1
        elif refs_in and not refs_out:
            qas_evidence_lost += 1
        episodic_clean.append({
            "question": q.get("question"),
            "answer": q.get("answer"),
            "image_ref": refs_out,
            "detail_type": bool(q.get("detail_type", False)),
        })

    qa = {"semantic_qa": semantic_clean, "episodic_qa": episodic_clean}
    stats = {
        "semantic_qa_count": len(semantic_clean),
        "episodic_qa_count": len(episodic_clean),
        "episodic_qa_no_evidence": qas_no_evidence,
        "episodic_qa_evidence_lost": qas_evidence_lost,
        "episodic_image_refs_dropped": refs_dropped,
    }
    return qa, stats


def _copy_one(src: Path, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "skip"
    shutil.copy2(src, dst)
    return "copy"


def copy_images(src_dir: Path, dst_dir: Path, workers: int = 16) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in src_dir.iterdir() if p.is_file()]
    if workers <= 1:
        for p in files:
            _copy_one(p, dst_dir / p.name)
        return len(files)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_copy_one, p, dst_dir / p.name) for p in files]
        for fut in as_completed(futures):
            fut.result()
    return len(files)


def process_user(uid: int, copy_imgs: bool, workers: int = 16) -> dict:
    src = SRC_ROOT / str(uid)
    dst = OUT_ROOT / f"u{uid}"
    dst.mkdir(parents=True, exist_ok=True)

    info_dir = src / "info_to_mada"
    src_csv = find_user_csv(info_dir)
    dst_csv = dst / "photos.csv"
    nsid, nickname = write_english_csv(src_csv, dst_csv)
    years, csv_rows = parse_year_range(dst_csv)

    qa, qa_stats = build_qa(src / "album_data.json", src / "images")
    with (dst / "qa.json").open("w") as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)

    profile_src = src / "profile.jpg"
    if profile_src.exists():
        shutil.copy2(profile_src, dst / "profile.jpg")

    img_count_src = sum(1 for _ in (src / "images").iterdir())
    if copy_imgs:
        img_count_out = copy_images(src / "images", dst / "images", workers=workers)
    else:
        img_count_out = -1

    return {
        "id": f"u{uid}",
        "nsid": nsid,
        "nickname": nickname,
        "years": years,
        "photo_count": img_count_src,
        "csv_rows": csv_rows,
        **qa_stats,
        "images_copied": img_count_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="all",
                    help="comma-separated user ids, or 'all' (default)")
    ap.add_argument("--no-images", action="store_true",
                    help="skip copying image bytes (fast metadata-only run)")
    ap.add_argument("--workers", type=int, default=16,
                    help="threads for image copy per user (default: 16)")
    args = ap.parse_args()

    if args.users == "all":
        uids = list(range(1, 21))
    else:
        uids = [int(x) for x in args.users.split(",") if x.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"writing release tree to: {OUT_ROOT}")
    print(f"users: {uids}")
    print(f"copy images: {not args.no_images}")
    print()

    summary = []
    for uid in uids:
        print(f"--- u{uid} ---", flush=True)
        try:
            stat = process_user(uid, copy_imgs=not args.no_images, workers=args.workers)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise
        for k, v in stat.items():
            print(f"  {k}: {v}", flush=True)
        summary.append(stat)
        print(flush=True)

    manifest = {"users": [
        {
            "id": s["id"],
            "nsid": s["nsid"],
            "nickname": s["nickname"],
            "years": s["years"],
            "photoCount": s["photo_count"],
            "semanticQaCount": s["semantic_qa_count"],
            "episodicQaCount": s["episodic_qa_count"],
        }
        for s in summary
    ]}
    if args.users == "all":
        with (OUT_ROOT / "users.json").open("w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"wrote {OUT_ROOT / 'users.json'}")

    print("\n=== summary ===")
    print(f"{'user':<5} {'nickname':<25} {'years':<14} {'photos':>7} {'sem_qa':>7} {'epi_qa':>7} {'no_evi':>7} {'lost':>5}")
    for s in summary:
        print(f"{s['id']:<5} {(s['nickname'] or '')[:24]:<25} {(s['years'] or '')[:13]:<14} "
              f"{s['photo_count']:>7} {s['semantic_qa_count']:>7} {s['episodic_qa_count']:>7} "
              f"{s['episodic_qa_no_evidence']:>7} {s['episodic_qa_evidence_lost']:>5}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate the release/ tree before publishing to HuggingFace.

Checks per user:
  - directory exists
  - profile.jpg, qa.json, photos.csv present
  - images/ has at least 1 file
  - qa.json has 'semantic_qa' and 'episodic_qa' keys
  - every image_ref in episodic_qa resolves to a file in images/
  - photos.csv has the expected English-only header
  - reports total photo / QA counts

Exit code is 0 if everything is clean, 1 otherwise.
"""

import csv
import json
import os
import sys
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "release"

EXPECTED_CSV_HEADER = [
    "photoid", "uid", "unickname", "datetaken", "dateuploaded", "capturedevice",
    "title", "description", "usertags", "machinetags",
    "longitude", "latitude", "accuracy",
    "pageurl", "downloadurl", "licensename", "licenseurl",
    "serverid", "farmid", "secret", "secretoriginal", "ext", "marker",
]


def check_user(udir: Path) -> tuple[list[str], dict]:
    errs = []
    stats = {}

    if not udir.is_dir():
        return [f"{udir}: not a directory"], stats

    for required in ["profile.jpg", "qa.json", "photos.csv", "images"]:
        if not (udir / required).exists():
            errs.append(f"{udir.name}: missing {required}")

    img_dir = udir / "images"
    images = {p.name for p in img_dir.iterdir() if p.is_file()} if img_dir.is_dir() else set()
    stats["n_images"] = len(images)
    if not images:
        errs.append(f"{udir.name}: images/ is empty")

    qa_path = udir / "qa.json"
    if qa_path.exists():
        with qa_path.open() as f:
            qa = json.load(f)
        sem = qa.get("semantic_qa")
        epi = qa.get("episodic_qa")
        if not isinstance(sem, list):
            errs.append(f"{udir.name}: qa.json missing semantic_qa list")
        if not isinstance(epi, list):
            errs.append(f"{udir.name}: qa.json missing episodic_qa list")
        stats["n_semantic"] = len(sem) if isinstance(sem, list) else 0
        stats["n_episodic"] = len(epi) if isinstance(epi, list) else 0

        for i, q in enumerate(epi or []):
            for ref in q.get("image_ref", []) or []:
                if ref not in images:
                    errs.append(f"{udir.name}: episodic_qa[{i}] image_ref '{ref}' not in images/")

    csv_path = udir / "photos.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            header = next(csv.reader(f))
            n_rows = sum(1 for _ in f)
        if header != EXPECTED_CSV_HEADER:
            errs.append(f"{udir.name}: photos.csv header mismatch\n  got:      {header}\n  expected: {EXPECTED_CSV_HEADER}")
        for col in header:
            if col.endswith("_zh"):
                errs.append(f"{udir.name}: photos.csv still has Chinese column {col!r}")
        stats["n_csv_rows"] = n_rows

    return errs, stats


def main() -> int:
    if not OUT_ROOT.is_dir():
        print(f"FATAL: release dir not found: {OUT_ROOT}", file=sys.stderr)
        return 1

    for top in ["README.md", "LICENSE", "users.json"]:
        if not (OUT_ROOT / top).exists():
            print(f"WARN: top-level {top} missing")

    user_dirs = sorted([p for p in OUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("u")],
                       key=lambda p: int(p.name[1:]))
    print(f"validating {len(user_dirs)} users under {OUT_ROOT}")
    print()

    all_errs = []
    totals = {"n_images": 0, "n_semantic": 0, "n_episodic": 0, "n_csv_rows": 0}
    print(f"{'user':<5} {'images':>7} {'sem':>4} {'epi':>4} {'csv':>5}  errors")
    for udir in user_dirs:
        errs, stats = check_user(udir)
        for k in totals:
            totals[k] += stats.get(k, 0)
        print(f"{udir.name:<5} {stats.get('n_images', 0):>7} {stats.get('n_semantic', 0):>4} "
              f"{stats.get('n_episodic', 0):>4} {stats.get('n_csv_rows', 0):>5}  {len(errs)}")
        all_errs.extend(errs)

    print()
    print(f"totals: {totals}")
    print()

    if all_errs:
        print(f"FAILED with {len(all_errs)} error(s):")
        for e in all_errs[:30]:
            print(f"  {e}")
        if len(all_errs) > 30:
            print(f"  ... and {len(all_errs) - 30} more")
        return 1

    print("OK — release tree validates clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

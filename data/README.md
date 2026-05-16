# camroll data

This folder is the build root for the **camroll-yfcc20** dataset — the
ground-truth benchmark that powers the [camroll](../README.md) demo.

The dataset itself is published on HuggingFace; no image bytes live in
this Git repo.

> 📦 **HuggingFace:** [`thaoshibe/camroll-yfcc20`](https://huggingface.co/datasets/thaoshibe/camroll-yfcc20)

## What's here

```
data/
├── README.md           ← you are here
├── prep_release.py     ← build script: source data → release/ tree
└── release/            ← generated (git-ignored), pushed to HuggingFace
    ├── README.md
    ├── LICENSE
    ├── users.json
    └── u{1..20}/
        ├── images/
        ├── profile.jpg
        ├── qa.json
        └── photos.csv
```

## Dataset at a glance

| | |
|---|---|
| Users | 20 anonymized Flickr photographers |
| Photos | ~15,000 original YFCC100M JPEGs (~1.9 GB) |
| QA pairs | 1,000 total — 200 *semantic* + 800 *episodic* |
| Languages | English (annotations) |
| Licenses | Per-photo CC (varies); annotations CC BY 4.0 |

See [`release/README.md`](release/README.md) for the full schema, loading
instructions, license details, and citation.

## Rebuilding from source

```bash
cd camroll/data
python3 prep_release.py             # full build, ~30 min, ~1.9 GB
python3 prep_release.py --no-images # metadata-only, ~20 s
python3 prep_release.py --users 1   # single user
```

Source data lives under
`/sensei-fs-3/tenants/Sensei-AdobeResearchTeam/thaon/code/thaodata/yfcc_v3/`
(internal). The build script normalizes image references, URL-decodes
text fields, drops Chinese duplicates, and emits a clean release tree.

## Publishing to HuggingFace

```bash
cd release
huggingface-cli login
huggingface-cli upload thaoshibe/camroll-yfcc20 . . \
    --repo-type=dataset --commit-message="initial release"
```

# camroll demo

A static demo of the camroll project — pick a YFCC100M user, browse their
photo roll, click suggested questions to see real agent traces with the
relevant photos highlighted on the left.

## What's in this folder

```
camroll/
├── camroll_demo.html       ← the demo page (open this in a browser)
├── yfcc_users.json         ← lightweight manifest used by the picker
├── yfcc_users/
│   ├── u1.json … u14.json  ← full per-user data (photos + events + agent traces)
│   └── u1.jpg  … u14.jpg   ← 200×200 profile photos (~10 KB each)
└── README.md
```

14 YFCC users, ~11,000 photos total. The HTML loads `yfcc_users.json` first
(small, instant), then lazy-loads `yfcc_users/u{N}.json` (~500 KB – 1 MB)
only when a user is clicked. Photo bytes themselves are served by Flickr —
nothing is committed to the repo.

## Run locally

```bash
cd camroll
python3 -m http.server 8765
# → http://localhost:8765/camroll_demo.html
```

## Deploy to GitHub Pages

1. Put this `camroll/` folder at the root of your `thaoshibe.github.io`
   repo, commit, push.
2. Pages will serve it at
   `https://thaoshibe.github.io/camroll/camroll_demo.html`.

If you instead push this folder to a dedicated repo named `camroll` (in
your `thaoshibe` account), it would live at
`https://thaoshibe.github.io/camroll/camroll_demo.html` as well, since
GitHub Pages mounts project sites at `<user>.github.io/<repo>/`.

## Data shape

### `yfcc_users.json` (manifest)

```jsonc
{
  "users": [
    {
      "id": "u1",
      "name": "alaspoorwho",
      "years": "2003 – 2013",
      "photoCount": 827,
      "eventCount": 92,
      "avatar":   "yfcc_users/u1.jpg",
      "preview":  ["https://…_q.jpg", "…", "…", "…"],  // 4 thumbs for picker mosaic
      "dataFile": "yfcc_users/u1.json"
    }
  ]
}
```

### `yfcc_users/u{N}.json` (per-user)

```jsonc
{
  "id": "u1",
  "name": "alaspoorwho",
  "nsid": "12028361_N00",
  "flickrProfile": "https://www.flickr.com/photos/12028361@N00/",
  "years": "2003 – 2013",
  "photoCount": 827,
  "avatar": "yfcc_users/u1.jpg",
  "events": [                       // kii event groupings (not currently rendered, but available)
    {"id":"e0","name":"…","date":"2003-11-27","description":"…","photoIds":["…"]}
  ],
  "suggestedQs": [
    {
      "question":    "What color was the Smart car…?",
      "qType":       "episodic",
      "answer":      "Red",
      "explanation": "Judge's short explanation of why this is correct.",
      "highlight":   ["348235464"],           // photo ids to outline + scroll-to
      "latencyS":    6.713,
      "trace": [
        {"tool":"search_captions","query":"…","snippet":"<full tool result>","score":0.59},
        ...
      ]
    }
  ],
  "photos": [
    {
      "id":      "8638574342",
      "thumb":   "https://farm9.staticflickr.com/8121/8638574342_8d1eabf693_q.jpg",
      "src":     "https://farm9.staticflickr.com/8121/8638574342_8d1eabf693_b.jpg",
      "page":    "https://www.flickr.com/photos/12028361@N00/8638574342/",
      "date":    "2013-04-09T18:26:54",
      "caption":     "<original Flickr title>",
      "captionKii":  "<richer kii first-person caption>",
      "license":     "Attribution-NonCommercial-ShareAlike License",
      "licenseUrl":  "https://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
  ]
}
```

Photos are sorted newest-first (Google Photos style). The demo groups
consecutive photos from the same month under a "Month YYYY" sticky header.
Suggested questions are picked from agent eval traces that scored 10/10
from the LLM judge — so every demo question is one the agent answered
correctly.

## Privacy / data handling

- Nothing the visitor types is saved, transmitted, or logged. The chat is
  fully client-side; free-form questions return a polite "demo-only" message.
- Photos are served directly by Flickr — each lightbox shows the original
  Flickr page link and the CC license, satisfying attribution.

## License

Photos remain under their original Creative Commons licenses (each photo's
lightbox shows the exact one + a link to the legal text). The demo HTML/JS
is your own.

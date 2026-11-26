# Test data

## Web recordings (`recorder/`)

Real browser sessions recorded with Playwright, for testing the video pipeline on
much more variety than the synthetic scenes.

```sh
cd testdata/recorder && npm install
node record.mjs                 # all scenarios (site ones need network)
node record.mjs "wiki|github"   # a subset (regex on names)
```

Each scenario writes to `samples/web/` (gitignored, ~60 MB, regenerate any time):

- `NAME.mp4`: the recording (viewport-sized, 25 fps, with a visible mouse cursor)
- `NAME.truth.jsonl`: ground truth, ~1 line/second: `{t, url, scrollY, items: [{text, boxes: [[x,y,w,h], …]}]}`,
  the visible DOM text with its on-screen boxes, timestamped against the video
- `NAME.json`: what the scenario is (viewport, theme, site/local, canvas, description)

Scenarios mix **real sites** (Wikipedia desktop/mobile/dark/table, Hacker News,
GitHub repo + code view, MDN dark, Python docs, Excalidraw drawing, OpenStreetMap,
NPR, arXiv, Wikipedia search) and **local app pages** in `recorder/pages/`
(dashboard light/dark, kanban drag-and-drop, form validation + modal, slides,
IDE typing + terminal, chat, spreadsheet edits/sort, diagram editor, dark inbox,
settings toggles/menus, docs scroll + 150% zoom, app switching, video call with
shared slides).

Interactions: scrolling, clicking links/tabs, typing, dragging, hover,
dropdowns, modals, page navigations, live-updating content, animation.

Caveats on the truth: text drawn on a `<canvas>` (Excalidraw, map tiles) has no
DOM text, so those are flagged `canvas` and not scored; GitHub's code view
exposes little of its code as DOM text.

## Scoring

```sh
cd pipeline && uv run python -m replicator.webeval [name-regex] [--interval 2]
```

Runs the video pipeline on each recording (output viewable in the viewer as
`▶ video/web-NAME`) and compares each processed frame's board text with the DOM
truth at that moment, as bags of words (recall / precision), plus timing.

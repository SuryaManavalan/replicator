# replicator

Screenshot → whiteboard. Turns screen content into a text **board syntax** that an LLM can read and our Excalidraw-based viewer can render.

- `syntax/SPEC.md` — the board syntax (v0, iterating).
- `pipeline/` — Python: board model/serializer, synthetic screenshot generator (vision pipeline next).
- `samples/synthetic/` — generated `NAME.png` + ground-truth `NAME.board` pairs.
- `viewer/` — React + Excalidraw: source image, live-editable syntax, rendered board.

## Run

```sh
# regenerate synthetic samples
cd pipeline && uv run python -m replicator.synth

# viewer
cd viewer && npm install && npm run dev
npm run check   # parse all samples
```

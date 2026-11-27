"""Score the video pipeline on the recorded web test set (samples/web/).

Each recording has DOM ground truth (visible text, sampled ~1/s). For every
frame the pipeline processes, we compare the board's text with the truth
snapshot taken at the same moment, as bags of words — robust to the pipeline
splitting/merging lines differently than the DOM does.

Usage:  uv run python -m replicator.webeval [name-regex] [--interval 2]
Writes samples/video/web-NAME/ (viewable in the viewer) and prints a table;
summary also saved to samples/web/scores.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .board import Shape, Table, Text
from .diff import norm
from .video import run

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "samples" / "web"
OUT = ROOT / "samples" / "video"

WORD = re.compile(r"[A-Za-z0-9$%#@][A-Za-z0-9$%#@.,:'/+-]*")


def words(text: str) -> Counter:
    c = Counter()
    for w in WORD.findall(text):
        n = norm(w)
        if len(n) >= 2:
            c[n] += 1
    return c


QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
EDGE = re.compile(r'^\s*(?:"(?:[^"\\]|\\.)*"|#[\w-]+)\s+(?:->|<-|<->|--)\s')


def board_words(path: Path) -> Counter:
    """Words shown on a serialized board: quoted text, except that edge lines
    only contribute their own label (their endpoints repeat node labels)."""
    c = Counter()
    for line in path.read_text().splitlines():
        if EDGE.match(line):
            line = line.split(" : ", 1)[1] if " : " in line else ""
        for s in QUOTED.findall(line):
            c.update(words(s.replace("\\n", " ")))
    return c


def truth_at(truth: list[dict], t: float, tol: float = 0.75) -> Counter | None:
    best = min(truth, key=lambda s: abs(s["t"] - t), default=None)
    if not best or abs(best["t"] - t) > tol:
        return None
    c = Counter()
    for item in best["items"]:
        c.update(words(item["text"]))
    return c


def score(name: str, interval: float, rerun: bool = True) -> dict:
    meta = json.loads((WEB / f"{name}.json").read_text())
    truth = [json.loads(l) for l in (WEB / f"{name}.truth.jsonl").read_text().splitlines() if l.strip()]
    out = OUT / f"web-{name}"
    if rerun or not (out / "steps.json").exists():
        res = run(WEB / f"{name}.mp4", interval, out, gate=0.0003, width=1000, verbose=False)
        all_steps, n_events = res["steps"], res["events"]
    else:
        all_steps = json.loads((out / "steps.json").read_text())
        n_events = (out / "events.log").read_text().count("\n@") + 1
    steps = [s for s in all_steps if s.get("frame")]
    boards = {s["frame"]: board_words(out / "boards" / f"{s['frame']:04d}.board") for s in steps}

    def at(offset):
        rh = rt = ph = pt = 0
        for s in steps:
            tw = truth_at(truth, s["t"] + offset)
            if tw is None:
                continue
            bw = boards[s["frame"]]
            both = sum((tw & bw).values())
            rh, rt, ph, pt = rh + both, rt + sum(tw.values()), ph + both, pt + sum(bw.values())
        return rh, rt, ph, pt

    # The truth clock starts at page creation; the video's first frame isn't
    # exactly then. Fit ONE offset per video (same for every frame).
    offsets = [o / 4 for o in range(-8, 9)]
    best = max(offsets, key=lambda o: (lambda r: r[0] / r[1] if r[1] else 0)(at(o)))
    rh, rt, ph, pt = at(best)
    r0 = at(0.0)
    ms = [s["ms_total"] + s.get("confirm", {}).get("ms_total", 0) for s in steps]
    return {
        "name": name, "kind": meta["kind"], "theme": meta["theme"], "viewport": meta["viewport"],
        "canvas": meta["canvas"], "offset": best,
        "recall": round(rh / rt, 3) if rt else None,
        "precision": round(ph / pt, 3) if pt else None,
        "recall_unaligned": round(r0[0] / r0[1], 3) if r0[1] else None,
        "steps": len(all_steps), "processed": len(steps), "events": n_events,
        "avg_ms": round(sum(ms) / len(ms)) if ms else None, "max_ms": max(ms) if ms else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--rescore", action="store_true", help="score existing outputs without rerunning the pipeline")
    args = ap.parse_args()
    names = sorted(p.stem for p in WEB.glob("*.mp4") if re.search(args.filter, p.stem))
    rows = []
    print(f"{'video':22} {'theme':5} {'viewport':11} {'recall':>7} {'prec':>6} {'(raw R)':>8} {'off':>5} {'steps':>6} {'events':>6} {'avg ms':>7} {'max ms':>7}")
    for n in names:
        try:
            r = score(n, args.interval, rerun=not args.rescore)
        except Exception as e:  # one broken video shouldn't stop the run
            print(f"{n:22} ERROR {type(e).__name__}: {e}", flush=True)
            rows.append({"name": n, "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(r)
        f = lambda v: "  -  " if v is None else f"{v:.2f}"
        vp = "x".join(map(str, r["viewport"]))
        note = " (canvas)" if r["canvas"] else ""
        print(f"{n:22} {r['theme']:5} {vp:11} {f(r['recall']):>7} {f(r['precision']):>6} {f(r['recall_unaligned']):>8} {r['offset']:>+5.2f} {r['processed']:>3}/{r['steps']:<2} "
              f"{r['events']:>6} {r['avg_ms'] or 0:>7} {r['max_ms'] or 0:>7}{note}", flush=True)
    scored = [r for r in rows if r.get("recall") is not None and not r["canvas"]]
    if scored:
        print(f"\nmean recall {sum(r['recall'] for r in scored) / len(scored):.3f}   "
              f"mean precision {sum(r['precision'] for r in scored) / len(scored):.3f}   "
              f"mean step {sum(r['avg_ms'] for r in scored) / len(scored):.0f} ms   (non-canvas, {len(scored)} videos)")
    (WEB / "scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

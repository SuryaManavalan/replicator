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


def board_words(path: Path) -> Counter:
    """Words in a serialized board: everything inside quotes."""
    c = Counter()
    for s in re.findall(r'"((?:[^"\\]|\\.)*)"', path.read_text()):
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


def score(name: str, interval: float) -> dict:
    meta = json.loads((WEB / f"{name}.json").read_text())
    truth = [json.loads(l) for l in (WEB / f"{name}.truth.jsonl").read_text().splitlines() if l.strip()]
    out = OUT / f"web-{name}"
    res = run(WEB / f"{name}.mp4", interval, out, gate=0.0003, width=1000, verbose=False)
    steps = [s for s in res["steps"] if s.get("frame")]
    rec_hit = rec_tot = pre_hit = pre_tot = 0
    for s in steps:
        tw = truth_at(truth, s["t"])
        if tw is None:
            continue
        bw = board_words(out / "boards" / f"{s['frame']:04d}.board")
        both = sum((tw & bw).values())
        rec_hit += both
        rec_tot += sum(tw.values())
        pre_hit += both
        pre_tot += sum(bw.values())
    ms = [s["ms_total"] + s.get("confirm", {}).get("ms_total", 0) for s in steps]
    return {
        "name": name, "kind": meta["kind"], "theme": meta["theme"], "viewport": meta["viewport"],
        "canvas": meta["canvas"],
        "recall": round(rec_hit / rec_tot, 3) if rec_tot else None,
        "precision": round(pre_hit / pre_tot, 3) if pre_tot else None,
        "steps": len(res["steps"]), "processed": len(steps), "events": res["events"],
        "avg_ms": round(sum(ms) / len(ms)) if ms else None, "max_ms": max(ms) if ms else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    names = sorted(p.stem for p in WEB.glob("*.mp4") if re.search(args.filter, p.stem))
    rows = []
    print(f"{'video':22} {'theme':5} {'viewport':11} {'recall':>7} {'prec':>6} {'steps':>6} {'events':>6} {'avg ms':>7} {'max ms':>7}")
    for n in names:
        r = score(n, args.interval)
        rows.append(r)
        f = lambda v: "  -  " if v is None else f"{v:.2f}"
        vp = "x".join(map(str, r["viewport"]))
        note = " (canvas)" if r["canvas"] else ""
        print(f"{n:22} {r['theme']:5} {vp:11} {f(r['recall']):>7} {f(r['precision']):>6} {r['processed']:>3}/{r['steps']:<2} "
              f"{r['events']:>6} {r['avg_ms'] or 0:>7} {r['max_ms'] or 0:>7}{note}", flush=True)
    scored = [r for r in rows if r["recall"] is not None and not r["canvas"]]
    if scored:
        print(f"\nmean recall {sum(r['recall'] for r in scored) / len(scored):.3f}   "
              f"mean precision {sum(r['precision'] for r in scored) / len(scored):.3f}   "
              f"mean step {sum(r['avg_ms'] for r in scored) / len(scored):.0f} ms   (non-canvas, {len(scored)} videos)")
    (WEB / "scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

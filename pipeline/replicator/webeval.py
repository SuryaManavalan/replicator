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


def _unq(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), s)


def board_blocks(path: Path) -> tuple[list[dict], list[dict]]:
    """Tables and code blocks from a serialized board."""
    tables, code, cur = [], [], None
    for line in path.read_text().splitlines():
        body = line.lstrip()
        ind = len(line) - len(body)
        m = re.match(r"(table|code)\s+@(-?\d+),(-?\d+)\s+(\d+)x(\d+)(.*)", body)
        if m:
            box = tuple(int(v) for v in m.groups()[1:5])
            if m.group(1) == "table":
                cur = {"box": box, "rows": [], "ind": ind}
                tables.append(cur)
            else:
                st = re.search(r"from (\d+)", m.group(6))
                cur = {"box": box, "lines": [], "start": int(st.group(1)) if st else None, "ind": ind}
                code.append(cur)
            continue
        if cur is not None and ind > cur["ind"] and body.startswith('"'):
            cells = [_unq(c) for c in QUOTED.findall(body)]
            if "rows" in cur:
                cur["rows"].append(cells)
            else:
                cur["lines"].append(cells[0] if cells else "")
            continue
        cur = None
    return tables, code


def _iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i) if i else 0.0


def table_cells(truth_tables: list[dict], pred_tables: list[dict], scale: float) -> tuple[int, int]:
    """Cells exactly right / truth cells, for tables fully on screen. Rows are
    aligned by the offset that matches best (a table can be partly scrolled)."""
    hit = tot = 0
    for t in truth_tables:
        rows = [r["cells"] for r in t["rows"]]  # the recorder keeps only on-screen rows
        if not t.get("fully_visible"):
            # partly scrolled: score rows wholly on screen, against their own box
            vis = [r for r in t["rows"] if r["box"][1] >= 0 and r["box"][1] + r["box"][3] <= 10_000]
            if len(vis) < 3:
                continue
            rows = [r["cells"] for r in vis]
            x0 = min(r["box"][0] for r in vis); y0 = min(r["box"][1] for r in vis)
            x1 = max(r["box"][0] + r["box"][2] for r in vis); y1 = max(r["box"][1] + r["box"][3] for r in vis)
            t = dict(t, box=[x0, y0, x1 - x0, y1 - y0])
        tot += sum(len(r) for r in rows)
        tb = [v * scale for v in t["box"]]
        p = max(pred_tables, key=lambda p: _iou(tb, p["box"]), default=None)
        if p is None or _iou(tb, p["box"]) < 0.2:
            continue
        best = 0
        # rows can be offset (partly scrolled); columns too (on-screen columns the
        # DOM text doesn't have, e.g. CSS-generated rank numbers)
        for off in range(-4, 5):
            for co in (0, 1, -1):
                h = sum(1 for i, r in enumerate(rows) for j, v in enumerate(r)
                        if 0 <= i + off < len(p["rows"]) and 0 <= j + co < len(p["rows"][i + off])
                        and norm(p["rows"][i + off][j + co]) == norm(v))
                best = max(best, h)
        hit += best
    return hit, tot


def code_lines(truth_code: list[dict], pred_code: list[dict], scale: float) -> tuple[int, int]:
    """Code lines exactly right (indentation included) / truth lines."""
    hit = tot = 0
    for t in truth_code:
        if not t.get("fully_visible"):
            continue
        lines = t["lines"]
        # editors that put line numbers inside the <pre>: strip them from the truth
        if all(re.match(rf"\s*{i + 1}(?!\d)", l) for i, l in enumerate(lines[:3])):
            lines = [re.sub(rf"^\s*{i + 1}", "", l, count=1) for i, l in enumerate(lines)]
        lines = [l.rstrip() for l in lines]
        tot += len(lines)
        tb = [v * scale for v in t["box"]]
        p = max(pred_code, key=lambda p: _iou(tb, p["box"]), default=None)
        if p is None or _iou(tb, p["box"]) < 0.1:
            continue
        pl = [l.rstrip() for l in p["lines"]]
        best = 0
        for off in range(-2, 3):
            best = max(best, sum(1 for i, l in enumerate(lines) if 0 <= i + off < len(pl) and pl[i + off] == l))
        hit += best
    return hit, tot


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
    # structure, at the same fitted offset
    scale = 1000 / meta["viewport"][0]
    cells = [0, 0]
    lines = [0, 0]
    for s_ in steps:
        snap = min(truth, key=lambda x: abs(x["t"] - (s_["t"] + best)), default=None)
        if not snap or abs(snap["t"] - (s_["t"] + best)) > 0.75:
            continue
        pt_, pc_ = board_blocks(out / "boards" / f"{s_['frame']:04d}.board")
        h, t_ = table_cells(snap.get("tables", []), pt_, scale)
        cells[0] += h; cells[1] += t_
        h, t_ = code_lines(snap.get("code", []), pc_, scale)
        lines[0] += h; lines[1] += t_
    ms = [s["ms_total"] + s.get("confirm", {}).get("ms_total", 0) for s in steps]
    return {
        "name": name, "kind": meta["kind"], "theme": meta["theme"], "viewport": meta["viewport"],
        "canvas": meta["canvas"], "offset": best,
        "recall": round(rh / rt, 3) if rt else None,
        "precision": round(ph / pt, 3) if pt else None,
        "recall_unaligned": round(r0[0] / r0[1], 3) if r0[1] else None,
        "table_cells": f"{cells[0]}/{cells[1]}" if cells[1] else None,
        "code_lines": f"{lines[0]}/{lines[1]}" if lines[1] else None,
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
    print(f"{'video':22} {'theme':5} {'viewport':11} {'recall':>7} {'prec':>6} {'(raw R)':>8} {'off':>5} {'steps':>6} {'events':>6} {'avg ms':>7} {'max ms':>7} {'cells':>9} {'code':>7}")
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
              f"{r['events']:>6} {r['avg_ms'] or 0:>7} {r['max_ms'] or 0:>7} {r['table_cells'] or '-':>9} {r['code_lines'] or '-':>7}{note}", flush=True)
    scored = [r for r in rows if r.get("recall") is not None and not r["canvas"]]
    if scored:
        print(f"\nmean recall {sum(r['recall'] for r in scored) / len(scored):.3f}   "
              f"mean precision {sum(r['precision'] for r in scored) / len(scored):.3f}   "
              f"mean step {sum(r['avg_ms'] for r in scored) / len(scored):.0f} ms   (non-canvas, {len(scored)} videos)")
    (WEB / "scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

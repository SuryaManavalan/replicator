"""Score the converter against synthetic ground truth.

Usage:  uv run python -m replicator.evaluate [-v]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

from .board import Board, Shape, Table, Text
from .convert import convert, tidy_board
from .synth import SCENES, draw_board

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "synthetic"


def walk(items):
    for n in items:
        yield n
        if isinstance(n, Shape):
            yield from walk(n.children)


def strings(b: Board) -> Counter:
    c = Counter()
    for n in walk(b.items):
        if isinstance(n, Text):
            c.update(n.text.split("\n"))
        elif isinstance(n, Table):
            c.update(v for row in n.rows for v in row if v)
        elif n.label:
            c.update(n.label.split("\n"))
    c.update(e.label for e in b.edges if e.label)
    return c


def iou(a, b):
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    i = ix * iy
    return i / (a.w * a.h + b.w * b.h - i) if i else 0


def name(s: Shape) -> str:
    if s.label:
        return s.label
    texts = [c.text for c in s.children if isinstance(c, Text)]
    return texts[0] if texts else f"@{round(s.x)},{round(s.y)}"


def score(truth: Board, pred: Board, verbose=False):
    ts, ps = strings(truth), strings(pred)
    text_hit = sum((ts & ps).values())
    text = (text_hit / max(sum(ts.values()), 1), text_hit / max(sum(ps.values()), 1))

    tsh = [n for n in walk(truth.items) if isinstance(n, Shape)]
    psh = [n for n in walk(pred.items) if isinstance(n, Shape)]
    match = {}
    for t in tsh:
        best = max(psh, key=lambda p: iou(t, p), default=None)
        need = 0.4 if max(t.w, t.h) < 40 else 0.7  # icons: a few px off is a big IoU hit
        if best is not None and iou(t, best) > need and best.kind == t.kind:
            match[id(t)] = best
    shape = (len(match) / max(len(tsh), 1), len(match) / max(len(psh), 1))

    def key(e, m):
        a, b = (m.get(id(e.src)), m.get(id(e.dst))) if m is not None else (e.src, e.dst)
        if a is None or b is None:
            return None
        return (id(a), id(b), e.kind) if e.kind != "--" else (frozenset((id(a), id(b))), "--")

    tkeys = Counter(k for e in truth.edges if (k := key(e, match)) is not None)
    pkeys = Counter(key(e, None) for e in pred.edges)
    edge_hit = sum((tkeys & pkeys).values())
    edge = (edge_hit / max(len(truth.edges), 1), edge_hit / max(len(pred.edges), 1))

    if verbose:
        miss_t = ts - ps
        extra_t = ps - ts
        if miss_t: print("    missing text:", dict(miss_t))
        if extra_t: print("    extra text:  ", dict(extra_t))
        missed = [name(t) for t in tsh if id(t) not in match]
        if missed: print("    missing shapes:", missed)
        if truth.edges:
            print(f"    edges: truth {len(truth.edges)}, pred {len(pred.edges)}, matched {edge_hit}")
    tables = (sum(isinstance(n, Table) for n in walk(truth.items)),
              sum(isinstance(n, Table) for n in walk(pred.items)))
    if verbose and tables[0] != tables[1]:
        print(f"    tables: truth {tables[0]}, pred {tables[1]}")
    extra = [f"{n.kind}@{round(n.x)},{round(n.y)}" for n in psh
             if not any(m is n for m in match.values())]
    if verbose and extra:
        print("    extra shapes:", extra[:12])
    return text, shape, edge


def main():
    verbose = "-v" in sys.argv
    print(f"{'scene':18} {'text R/P':>12} {'shapes R/P':>12} {'edges R/P':>12}")
    for name_, build in SCENES.items():
        truth = tidy_board(build())
        img = cv2.imread(str(SAMPLES / f"{name_}.png"))
        if img is None:
            draw_board(truth).save(SAMPLES / f"{name_}.png")
            img = cv2.imread(str(SAMPLES / f"{name_}.png"))
        pred = convert(img).board
        (t, s, e) = score(truth, pred, verbose)
        f = lambda p: f"{p[0]:.2f}/{p[1]:.2f}"
        print(f"{name_:18} {f(t):>12} {f(s):>12} {f(e) if truth.edges else '—':>12}")
        if verbose:
            print()


if __name__ == "__main__":
    main()

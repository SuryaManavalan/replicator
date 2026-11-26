"""Board -> board differences, as Δ lines (the video stream's updates).

    ~ "$48,210" -> "$49,105"                 text changed in place
    ~ "Pending" -> "Shipped" (row "#1041")   a table cell changed
    + rect "Write tests" @46,302 266x55      added (with where it is)
    + row "#1043" | "Robert Fox" | …         a table row added
    - "Fix login redirect"                   removed
    > rect "Dark mode" @680,340              moved (new position)
    + "Check stock" -> "In stock?"           edges added / removed

diff() returns None when so much changed that a fresh KEY (full snapshot)
is clearer than a list of changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .board import Board, Shape, Table, Text, _geom, _style, quote

SKIP = ("icon", "icons", "group", "image")  # layout/decoration: not worth announcing
SHAPES = ("rect", "ellipse", "diamond")  # a hover can turn a pill "rect" into an "ellipse": same thing


@dataclass
class Entry:
    kind: str  # text | rect | ellipse | diamond | image | row
    text: str  # what identifies it to a reader
    x: float
    y: float
    w: float
    h: float
    line: str  # how to print it when added
    cells: list[str] | None = None
    unsure: bool = False  # OCR flagged it [?]: don't announce it on its own

    @property
    def ref(self) -> str:
        if self.kind == "text":
            return quote(self.text)
        if self.kind == "row":
            return "row " + self.text
        return f"{self.kind} {quote(self.text)}" if self.text else f"{self.kind} @{round(self.x)},{round(self.y)}"


def entries(board: Board) -> list[Entry]:
    out: list[Entry] = []

    def walk(items):
        for n in items:
            if isinstance(n, Text):
                # one entry per line: OCR merges/splits wrapped lines differently frame to frame
                lines = n.text.split("\n")
                lh = n.h / len(lines)
                for i, ln in enumerate(lines):
                    e = Text(ln, n.x, n.y + i * lh, n.w, lh, n.style)
                    out.append(Entry("text", ln, e.x, e.y, e.w, e.h, f"{quote(ln)} {_geom(e)}{_style(n.style)}",
                                     unsure=bool(n.style.get("?"))))
            elif isinstance(n, Table):
                pitch = n.h / max(len(n.rows), 1)
                for i, row in enumerate(n.rows):
                    txt = " | ".join(quote(c) for c in row)
                    out.append(Entry("row", txt, n.x, n.y + i * pitch, n.w, pitch, f"row {txt}", list(row)))
            elif isinstance(n, Shape):
                if n.kind not in SKIP and (n.label or n.kind not in SHAPES):
                    label = n.label or ""
                    head = f"{n.kind} {quote(label)}" if label else n.kind
                    out.append(Entry(n.kind, label, n.x, n.y, n.w, n.h, f"{head} {_geom(n)}{_style(n.style)}"))
                walk(n.children)

    walk(board.items)
    return out


TEXTUAL = SHAPES + ("text",)  # a hovered tab gains a box: "Tab" and rect "Tab" are one thing
_CONFUSABLE = str.maketrans({"I": "l", "1": "l", "|": "l", "!": "l", "O": "o", "0": "o", "·": ".", "•": ".",
                             "’": "'", "‘": "'", "“": '"', "”": '"'})


def norm(text: str) -> str:
    """Text as OCR-stable as we can make it: confusable glyphs folded together,
    spacing and stray punctuation dropped."""
    t = text.translate(_CONFUSABLE).lower()
    return re.sub(r"[\s.,:;'\"`~^*_\-]+", "", t)


def _same_kind(a: Entry, b: Entry) -> bool:
    return a.kind == b.kind or (a.kind in TEXTUAL and b.kind in TEXTUAL)


def _iou(a: Entry, b: Entry) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    i = ix * iy
    return i / (a.w * a.h + b.w * b.h - i) if i else 0.0


def _centre_dist(a: Entry, b: Entry) -> float:
    return ((a.x + a.w / 2 - b.x - b.w / 2) ** 2 + (a.y + a.h / 2 - b.y - b.h / 2) ** 2) ** 0.5


def _edge_keys(board: Board) -> dict[str, str]:
    def ref(s):
        return quote(s.label) if s.label else (f"#{s.id}" if s.id else "?")

    out = {}
    for e in board.edges:
        if not (e.src.label and e.dst.label):
            continue
        line = f"{ref(e.src)} {e.kind} {ref(e.dst)}" + (f" : {quote(e.label)}" if e.label else "")
        out[line] = line + _style(e.style)
    return out


def diff(old: Board, new: Board, key_ratio: float = 0.5) -> list[str] | None:
    a, b = entries(old), entries(new)
    move_tol = 0.02 * new.w
    left, right = list(a), list(b)
    same, moved, changed = [], [], []

    # 1. same kind + same text: the same element (nearest one if repeated).
    #    Unlabelled shapes have no text to go on: matched by overlap in 2c.
    for e in list(right):
        if not e.text:
            continue
        cands = [o for o in left if _same_kind(o, e) and norm(o.text) == norm(e.text)]
        if not cands:
            continue
        o = min(cands, key=lambda o: _centre_dist(o, e))
        left.remove(o)
        right.remove(e)
        if _centre_dist(o, e) > move_tol and e.kind != "row":  # rows re-flow when one is inserted
            moved.append((o, e))
        else:
            same.append(e)

    # 2a. table rows: same row if most cells agree (rows shift when one is inserted)
    for e in [e for e in right if e.kind == "row"]:
        def agree(o):
            if o.kind != "row" or not o.cells or not e.cells or len(o.cells) != len(e.cells):
                return 0
            return sum(a == b for a, b in zip(o.cells, e.cells))
        cands = [o for o in left if agree(o) >= max(1, len(e.cells) // 2 + 1) - (len(e.cells) <= 2)]
        if cands:
            o = max(cands, key=agree)
            left.remove(o)
            right.remove(e)
            changed.append((o, e))

    # 2b. same kind in the same place, different text: changed in place
    for e in list(right):
        if not e.text:
            continue  # unlabelled: handled in 2c
        cands = [o for o in left if _same_kind(o, e) and e.kind != "row" and o.text and _iou(o, e) > 0.5]
        if not cands:
            continue
        o = max(cands, key=lambda o: _iou(o, e))
        left.remove(o)
        right.remove(e)
        changed.append((o, e))

    # 2c. unlabelled shapes: same if they overlap well; moved if same size nearby
    for e in [e for e in right if not e.text]:
        cands = [o for o in left if o.kind == e.kind and not o.text and _iou(o, e) > 0.7]
        if cands:
            o = max(cands, key=lambda o: _iou(o, e))
            left.remove(o)
            right.remove(e)
            if _centre_dist(o, e) > move_tol:
                moved.append((o, e))
            else:
                same.append(e)

    removed, added = left, right
    offsets: dict = {}
    for o, e in moved:
        k = (round((e.x - o.x) / 4), round((e.y - o.y) / 4))
        offsets[k] = offsets.get(k, 0) + 1
    scrolling = any(v >= 3 for v in offsets.values())
    if not scrolling and len(added) + len(removed) > max(8, key_ratio * max(len(a), len(b), 1)):
        return None
    # OCR-unsure text isn't announced appearing/disappearing/changing on its own
    added = [e for e in added if not e.unsure]
    removed = [o for o in removed if not o.unsure]
    changed = [(o, e) for o, e in changed if not (o.unsure or e.unsure) and norm(o.text) != norm(e.text)]

    lines: list[str] = []
    for o, e in changed:
        if o.text == e.text:
            continue
        if e.kind == "row" and o.cells and e.cells and len(o.cells) == len(e.cells):
            for oc, ec in zip(o.cells, e.cells):
                if oc != ec:
                    lines.append(f"~ {quote(oc)} -> {quote(ec)} (row {quote(e.cells[0])})")
        elif e.kind == "text":
            lines.append(f"~ {quote(o.text)} -> {quote(e.text)}")
        else:
            lines.append(f"~ {e.kind} {quote(o.text)} -> {quote(e.text)}")
    lines += [f"+ {e.line}" for e in added]
    lines += [f"- {o.ref}" for o in removed]
    # many things moving by (about) the same offset = a scroll: one line, not dozens
    tol = 0.006 * new.w
    clusters: list[list] = []
    for o, e in sorted(moved, key=lambda p: (p[1].y - p[0].y, p[1].x - p[0].x)):
        dx, dy = e.x - o.x, e.y - o.y
        c = clusters[-1] if clusters else None
        if c and abs(dy - c[0][2]) <= tol and abs(dx - c[0][1]) <= tol:
            c.append((o, e, dx, dy))
        else:
            clusters.append([(o, e, dx, dy)])
    for c in sorted(clusters, key=len, reverse=True):
        if len(c) >= 3:
            dx = sorted(p[2] for p in c)[len(c) // 2]
            dy = sorted(p[3] for p in c)[len(c) // 2]
            x0 = round(min(p[1].x for p in c))
            x1 = round(max(p[1].x + p[1].w for p in c))
            what = "up" if dy < 0 else "down" if dy > 0 else ("left" if dx < 0 else "right")
            lines.append(f"> scrolled {what} {round(abs(dy) or abs(dx))} ({len(c)} items, x {x0}-{x1})")
        else:
            lines += [f"> {e.ref if e.text else e.kind} @{round(e.x)},{round(e.y)}" for _, e, _, _ in c]

    ea, eb = _edge_keys(old), _edge_keys(new)
    lines += [f"+ {eb[k]}" for k in eb if k not in ea]
    lines += [f"- {k}" for k in ea if k not in eb]
    return lines

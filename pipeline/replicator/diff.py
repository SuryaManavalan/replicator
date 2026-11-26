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

from dataclasses import dataclass

from .board import Board, Shape, Table, Text, _geom, _style, quote

SKIP = ("icon", "icons", "group")  # layout/decoration: not worth announcing


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
                out.append(Entry("text", n.text, n.x, n.y, n.w, n.h, f"{quote(n.text)} {_geom(n)}{_style(n.style)}"))
            elif isinstance(n, Table):
                pitch = n.h / max(len(n.rows), 1)
                for i, row in enumerate(n.rows):
                    txt = " | ".join(quote(c) for c in row)
                    out.append(Entry("row", txt, n.x, n.y + i * pitch, n.w, pitch, f"row {txt}", list(row)))
            elif isinstance(n, Shape):
                if n.kind not in SKIP:
                    label = n.label or ""
                    head = f"{n.kind} {quote(label)}" if label else n.kind
                    out.append(Entry(n.kind, label, n.x, n.y, n.w, n.h, f"{head} {_geom(n)}{_style(n.style)}"))
                walk(n.children)

    walk(board.items)
    return out


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
        cands = [o for o in left if o.kind == e.kind and o.text == e.text]
        if not cands:
            continue
        o = min(cands, key=lambda o: _centre_dist(o, e))
        left.remove(o)
        right.remove(e)
        if _centre_dist(o, e) > move_tol and e.kind != "row":  # rows re-flow when one is inserted
            moved.append(e)
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
        cands = [o for o in left if o.kind == e.kind and e.kind != "row" and o.text and _iou(o, e) > 0.5]
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
            (moved if _centre_dist(o, e) > move_tol else same).append(e)

    removed, added = left, right
    if len(added) + len(removed) > max(8, key_ratio * max(len(a), len(b), 1)):
        return None

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
    lines += [f"> {e.ref if e.text else e.kind} @{round(e.x)},{round(e.y)}" for e in moved]

    ea, eb = _edge_keys(old), _edge_keys(new)
    lines += [f"+ {eb[k]}" for k in eb if k not in ea]
    lines += [f"- {k}" for k in ea if k not in eb]
    return lines

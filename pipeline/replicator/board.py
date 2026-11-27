"""Board model + serializer for the v0 board syntax (see syntax/SPEC.md)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

SHAPE_KINDS = ("rect", "ellipse", "diamond", "image", "icon", "icons", "group")


@dataclass
class Shape:
    kind: str  # rect | ellipse | diamond | image | icon | icons (a run of n icons) | group (invisible)
    x: float
    y: float
    w: float
    h: float
    label: str | None = None
    id: str | None = None
    children: list[Shape | Text | Table | Code] = field(default_factory=list)
    style: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class Table:
    """Text aligned in rows and columns. cols are absolute column x positions."""
    x: float
    y: float
    w: float
    h: float
    cols: list[float]
    rows: list[list[str]]
    style: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class Code:
    """A block of code: exact lines (indentation preserved). `start` is the
    first line number when the editor shows a line-number gutter."""
    x: float
    y: float
    w: float
    h: float
    lines: list[str]
    start: int | None = None
    style: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class Text:
    text: str
    x: float
    y: float
    w: float
    h: float
    style: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class Edge:
    src: Shape
    dst: Shape
    label: str | None = None
    kind: str = "->"  # -> <- <-> --
    style: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class Board:
    w: int
    h: int
    items: list[Shape | Text | Table | Code] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def scale_board(board: Board, width: int = 1000) -> Board:
    """Rescale all geometry (in place) so the board is `width` wide."""
    f = width / board.w

    def sc(n):
        n.x, n.y, n.w, n.h = n.x * f, n.y * f, n.w * f, n.h * f
        if isinstance(n, Table):
            n.cols = [c * f for c in n.cols]
            if "size" in n.style:
                n.style["size"] = max(1, round(float(n.style["size"]) * f))
        if isinstance(n, Shape):
            for c in n.children:
                sc(c)

    for n in board.items:
        sc(n)
    board.w, board.h = width, round(board.h * f)
    return board


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _geom(n) -> str:
    return f"@{round(n.x)},{round(n.y)} {round(n.w)}x{round(n.h)}"


def _style(style: dict) -> str:
    if not style:
        return ""
    # keys starting with "_" are private hints (e.g. how synth draws an icon)
    parts = [k if v is True else f"{k}={v}" for k, v in style.items() if v and not k.startswith("_")]
    return f" [{' '.join(parts)}]" if parts else ""


def _gaps(items, axis: str, min_gap: float) -> list[float]:
    """Cut positions along x or y where no item crosses and the gap is wide enough."""
    if axis == "x":
        spans = sorted((n.x, n.x + n.w) for n in items)
    else:
        spans = sorted((n.y, n.y + n.h) for n in items)
    cuts, end = [], spans[0][1]
    for lo, hi in spans[1:]:
        if lo - end >= min_gap:
            cuts.append((end + lo) / 2)
        end = max(end, hi)
    return cuts


def _split(items, axis: str, cuts: list[float]):
    parts = [[] for _ in range(len(cuts) + 1)]
    for n in items:
        c = (n.x + n.w / 2) if axis == "x" else (n.y + n.h / 2)
        parts[sum(c > k for k in cuts)].append(n)
    return [p for p in parts if p]


def layout_blocks(items, min_gap: float = 8.0, depth: int = 0):
    """Recursive XY-cut on item boxes: full-width bands top-to-bottom, then
    columns left-to-right. Returns a nested list ("blocks") in reading order."""
    if len(items) <= 2 or depth > 8:
        return _row_order(items)
    ycuts = _gaps(items, "y", min_gap)
    if ycuts:
        return [layout_blocks(p, min_gap, depth + 1) for p in _split(items, "y", ycuts)]
    xcuts = _gaps(items, "x", min_gap)
    if xcuts:
        return [layout_blocks(p, min_gap, depth + 1) for p in _split(items, "x", xcuts)]
    return _row_order(items)


def _flatten(blocks):
    for b in blocks:
        if isinstance(b, list):
            yield from _flatten(b)
        else:
            yield b


def reading_order(items, min_gap: float = 8.0):
    """Reading order that respects layout: bands, then columns, then rows."""
    return list(_flatten(layout_blocks(list(items), min_gap)))


def _row_order(items):
    """Sort into rows (top-to-bottom), each row left-to-right.

    An item joins the current row when its top or its centre lines up with the
    row's first item (within half the smaller height). Comparing against the
    first item, not the row's growing span, stops tall panels from swallowing
    everything beside them.
    """
    items = sorted(items, key=lambda n: n.y)
    rows: list[list] = []
    for n in items:
        if rows:
            a = rows[-1][0]
            tol = 0.5 * min(n.h, a.h)
            if abs(n.y - a.y) < tol or abs((n.y + n.h / 2) - (a.y + a.h / 2)) < tol:
                rows[-1].append(n)
                continue
        rows.append([n])
    return [n for row in rows for n in sorted(row, key=lambda m: m.x)]


def _walk(items):
    for n in items:
        yield n
        if isinstance(n, Shape):
            yield from _walk(n.children)


def _slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:24].strip("-")
    if not slug:
        return "shape"
    return slug if slug[0].isalpha() else f"n{slug}"


def serialize(board: Board, geometry: str = "full") -> str:
    """geometry: "full" (every element), "boxes" (shapes/tables/groups only,
    not text or icons: lighter for an LLM, still renderable approximately),
    or "none" (structure and text only)."""
    shapes = [n for n in _walk(board.items) if isinstance(n, Shape)]
    label_counts = Counter(s.label for s in shapes if s.label)
    referenced = {id(e.src) for e in board.edges} | {id(e.dst) for e in board.edges}

    # Edges refer to shapes by label when every endpoint has a unique label.
    # Otherwise every endpoint gets an #id (consistent: never a mix), derived
    # from the label when there is one ("#n3", "#api-gateway").
    endpoints = [s for s in shapes if id(s) in referenced]
    use_ids = any(s.id or not s.label or label_counts[s.label] > 1 for s in endpoints)
    used = {s.id for s in shapes if s.id}
    counters: Counter = Counter()
    for s in shapes:
        if s.id:
            continue
        if use_ids and id(s) in referenced:
            if s.label:
                base = _slug(s.label)
            else:
                counters[s.kind] += 1
                base = f"{s.kind}{counters[s.kind]}"
            cand, k = base, 1
            while cand in used:
                k += 1
                cand = f"{base}-{k}"
            s.id = cand
            used.add(cand)

    def ref(s: Shape) -> str:
        return f"#{s.id}" if s.id else quote(s.label)  # type: ignore[arg-type]

    lines = [f"board {board.w}x{board.h}", ""]

    def g(node) -> str:
        if geometry == "none":
            return ""
        if geometry == "boxes" and (isinstance(node, Text) or getattr(node, "kind", "") in ("icon", "icons")):
            return ""
        return " " + _geom(node)

    def emit(items, depth):
        for node in reading_order(items):
            pad = "  " * depth
            if isinstance(node, Text):
                lines.append(f"{pad}{quote(node.text)}{g(node)}{_style(node.style)}")
            elif isinstance(node, Code):
                head = f"{pad}code{g(node)}"
                if node.start is not None:
                    head += f" from {node.start}"
                lines.append(head + _style(node.style))
                for ln in node.lines:
                    lines.append(f"{pad}  {quote(ln)}")
            elif isinstance(node, Table):
                cols = ",".join(str(round(c)) for c in node.cols)
                lines.append(f"{pad}table{g(node)} cols {cols}{_style(node.style)}")
                for row in node.rows:
                    lines.append(f"{pad}  " + " | ".join(quote(c) for c in row))
            else:
                head = node.kind
                if node.kind == "icons":
                    head += f" {node.style.get('_n', 0)}"
                if node.id:
                    head += f" #{node.id}"
                if node.label:
                    head += f" {quote(node.label)}"
                lines.append(f"{pad}{head}{g(node)}{_style(node.style)}")
                emit(node.children, depth + 1)

    emit(board.items, 0)

    if board.edges:
        lines.append("")
        for e in board.edges:
            line = f"{ref(e.src)} {e.kind} {ref(e.dst)}"
            if e.label:
                line += f" : {quote(e.label)}"
            lines.append(line + _style(e.style))

    return "\n".join(lines) + "\n"

"""Board model + serializer for the v0 board syntax (see syntax/SPEC.md)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

SHAPE_KINDS = ("rect", "ellipse", "diamond")


@dataclass
class Shape:
    kind: str  # rect | ellipse | diamond
    x: float
    y: float
    w: float
    h: float
    label: str | None = None
    id: str | None = None
    children: list[Shape | Text] = field(default_factory=list)
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
    items: list[Shape | Text] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _geom(n) -> str:
    return f"@{round(n.x)},{round(n.y)} {round(n.w)}x{round(n.h)}"


def _style(style: dict) -> str:
    if not style:
        return ""
    parts = [k if v is True else f"{k}={v}" for k, v in style.items() if v]
    return f" [{' '.join(parts)}]" if parts else ""


def reading_order(items):
    """Sort into rows (top-to-bottom), each row left-to-right."""
    items = sorted(items, key=lambda n: n.y)
    rows: list[list] = []
    for n in items:
        if rows:
            row = rows[-1]
            top = min(m.y for m in row)
            min_h = min(m.h for m in row)
            if n.y < top + 0.5 * min(min_h, n.h):
                row.append(n)
                continue
        rows.append([n])
    return [n for row in rows for n in sorted(row, key=lambda m: m.x)]


def _walk(items):
    for n in items:
        yield n
        if isinstance(n, Shape):
            yield from _walk(n.children)


def serialize(board: Board) -> str:
    shapes = [n for n in _walk(board.items) if isinstance(n, Shape)]
    label_counts = Counter(s.label for s in shapes if s.label)
    referenced = {id(e.src) for e in board.edges} | {id(e.dst) for e in board.edges}

    # Assign ids only where the label can't serve as identifier.
    used = {s.id for s in shapes if s.id}
    n = 0
    for s in shapes:
        needs = id(s) in referenced and (not s.label or label_counts[s.label] > 1)
        if needs and not s.id:
            n += 1
            while f"s{n}" in used:
                n += 1
            s.id = f"s{n}"
            used.add(s.id)

    def ref(s: Shape) -> str:
        return s.id if s.id else quote(s.label)  # type: ignore[arg-type]

    lines = [f"board {board.w}x{board.h}", ""]

    def emit(items, depth):
        for node in reading_order(items):
            pad = "  " * depth
            if isinstance(node, Text):
                lines.append(f"{pad}{quote(node.text)} {_geom(node)}{_style(node.style)}")
            else:
                head = node.kind
                if node.id:
                    head += f" {node.id}"
                if node.label:
                    head += f" {quote(node.label)}"
                lines.append(f"{pad}{head} {_geom(node)}{_style(node.style)}")
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

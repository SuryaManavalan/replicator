"""Synthetic screenshot generator.

Each scene is built as a Board (ground truth), drawn to a PNG with PIL, and
serialized to the board syntax. The pair (png, .board) lets us check the
vision pipeline against a known answer.

Usage:  uv run python -m replicator.synth [outdir]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .board import Board, Edge, Shape, Text, serialize

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LINE_HEIGHT = 1.25
LABEL_SIZE = 20

# Excalidraw-ish palette; names are what the syntax uses.
STROKE = {"black": "#1e1e1e", "gray": "#adb5bd", "red": "#e03131", "blue": "#1971c2", "green": "#2f9e44"}
FILL = {
    "gray": "#e9ecef", "red": "#ffc9c9", "pink": "#fcc2d7", "violet": "#d0bfff", "blue": "#a5d8ff",
    "cyan": "#99e9f2", "teal": "#96f2d7", "green": "#b2f2bb", "yellow": "#ffec99", "orange": "#ffd8a8",
}

_fonts: dict = {}


def font(size: float, bold=False, scale=1):
    key = (round(size * scale), bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_BOLD if bold else FONT, key[0])
    return _fonts[key]


# ---------- builders ----------

def text(s: str, x, y, size=20, bold=False) -> Text:
    f = font(size, bold)
    lines = s.split("\n")
    w = max(f.getlength(line) for line in lines)
    style = {"bold": True} if bold else {}
    return Text(s, x, y, math.ceil(w), round(len(lines) * size * LINE_HEIGHT), style)


def shape(kind, label, x, y, w, h, children=(), **style) -> Shape:
    return Shape(kind, x, y, w, h, label=label, children=list(children), style=style)


def rect(label, x, y, w, h, children=(), **style):
    return shape("rect", label, x, y, w, h, children, **style)


def ellipse(label, x, y, w, h, **style):
    return shape("ellipse", label, x, y, w, h, **style)


def diamond(label, x, y, w, h, **style):
    return shape("diamond", label, x, y, w, h, **style)


# ---------- scenes ----------

def scene_flow() -> Board:
    recv = rect("Receive order", 80, 300, 180, 80, fill="blue")
    check = rect("Check stock", 340, 300, 180, 80)
    instock = diamond("In stock?", 600, 280, 180, 120, fill="yellow")
    ship = rect("Ship order", 900, 300, 180, 80, fill="green")
    notify = rect("Notify customer", 600, 560, 180, 80, fill="red")
    return Board(1280, 800, [
        text("Order Flow", 60, 40, 32, bold=True),
        recv, check, instock, ship, notify,
        text("SLA: ship within 24h", 900, 420, 18),
    ], [
        Edge(recv, check), Edge(check, instock),
        Edge(instock, ship, "yes"), Edge(instock, notify, "no"),
    ])


def scene_architecture() -> Board:
    web = rect("Web App", 60, 330, 180, 90, fill="blue")
    api = rect("API Gateway", 350, 190, 190, 80, fill="violet")
    auth = rect("Auth Service", 660, 190, 190, 80)
    orders = rect("Orders Service", 350, 420, 190, 80)
    billing = rect("Billing Service", 660, 420, 190, 80)
    backend = rect("Backend", 310, 120, 580, 560, [api, auth, orders, billing], dashed=True)
    redis = ellipse("Redis", 990, 175, 200, 110, fill="red")
    pg = ellipse("Postgres", 990, 405, 200, 110, fill="green")
    return Board(1280, 800, [
        text("System Architecture", 40, 30, 32, bold=True),
        web, backend, redis, pg,
        text("TODO: add rate limiting", 40, 720, 18),
    ], [
        Edge(web, api, "HTTPS"), Edge(api, auth, "verify JWT"), Edge(api, orders),
        Edge(orders, billing, "invoice"), Edge(billing, pg, "SQL"), Edge(auth, redis, "sessions"),
    ])


def scene_dashboard() -> Board:
    topbar = rect(None, 0, 0, 1280, 64, [
        text("Acme Analytics", 24, 18, 22, bold=True),
        text("Settings", 1060, 22, 18),
        text("Log out", 1170, 22, 18),
    ], fill="gray", stroke="gray")
    sidebar = rect(None, 0, 64, 220, 736, [
        text("Overview", 24, 96, 18, bold=True),
        text("Reports", 24, 140, 18),
        text("Customers", 24, 184, 18),
        text("Billing", 24, 228, 18),
        text("Help", 24, 740, 18),
    ], stroke="gray")
    cards = []
    for i, (title, value) in enumerate([("Revenue", "$48,210"), ("Active users", "1,204"), ("Churn", "2.1%")]):
        x = 260 + i * 330
        cards.append(rect(None, x, 160, 300, 140, [
            text(title, x + 20, 180, 18), text(value, x + 20, 220, 36, bold=True),
        ], stroke="gray"))
    cols = [280, 420, 680, 900]
    rows = [
        ("Order", "Customer", "Amount", "Status"),
        ("#1042", "Jane Cooper", "$120.00", "Shipped"),
        ("#1041", "Wade Warren", "$89.50", "Pending"),
        ("#1040", "Esther Howard", "$240.00", "Shipped"),
        ("#1039", "Cameron Lee", "$15.99", "Refunded"),
    ]
    cells = [text("Recent orders", 280, 360, 20, bold=True)]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cells.append(text(val, cols[c], 420 + r * 50, 16, bold=(r == 0)))
    table = rect(None, 260, 340, 960, 400, cells, stroke="gray")
    return Board(1280, 800, [topbar, sidebar, text("Overview", 260, 96, 30, bold=True), *cards, table], [])


def scene_kanban() -> Board:
    columns = [
        ("To Do", "yellow", ["Write onboarding docs", "Fix login redirect", "Add CSV export"]),
        ("In Progress", "blue", ["Payment retries", "Dark mode"]),
        ("Done", "green", ["Upgrade to Node 22", "Rate limiter", "Audit logging"]),
    ]
    items = [text("Sprint 14", 40, 30, 32, bold=True)]
    for i, (title, color, cards) in enumerate(columns):
        x = 40 + i * 410
        kids = [rect(c, x + 20, 160 + j * 90, 340, 70, fill=color) for j, c in enumerate(cards)]
        items.append(rect(title, x, 100, 380, 660, kids, fill="gray"))
    return Board(1280, 800, items, [])


def scene_mindmap() -> Board:
    center = ellipse("Product Launch", 520, 340, 240, 120, fill="violet")
    items, edges = [center], []
    branches = [
        ("Marketing", 120, 120, [("Blog post", 40, 240), ("Launch video", 220, 240)]),
        ("Engineering", 960, 120, [("API freeze", 900, 240), ("Load test", 1080, 240)]),
        ("Sales", 120, 600, [("Pricing page", 40, 480), ("Partner deck", 220, 480)]),
        ("Support", 960, 600, [("FAQ", 900, 480), ("On-call rota", 1080, 480)]),
    ]
    for name, x, y, leaves in branches:
        b = rect(name, x, y, 200, 70, fill="blue")
        items.append(b)
        edges.append(Edge(center, b, kind="--"))
        for leaf, lx, ly in leaves:
            l = rect(leaf, lx, ly, 150, 50)
            items.append(l)
            edges.append(Edge(b, l, kind="--"))
    return Board(1280, 800, items, edges)


SCENES = {
    "01-flow": scene_flow,
    "02-architecture": scene_architecture,
    "03-dashboard": scene_dashboard,
    "04-kanban": scene_kanban,
    "05-mindmap": scene_mindmap,
}


# ---------- drawing ----------

def border_point(s: Shape, dx: float, dy: float):
    """Point on s's outline in direction (dx, dy) from its center."""
    cx, cy, hw, hh = s.x + s.w / 2, s.y + s.h / 2, s.w / 2, s.h / 2
    if s.kind == "ellipse":
        t = 1 / math.hypot(dx / hw, dy / hh)
    elif s.kind == "diamond":
        t = 1 / (abs(dx) / hw + abs(dy) / hh)
    else:
        t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


def edge_points(e: Edge, gap=6):
    a, b = e.src, e.dst
    dx = (b.x + b.w / 2) - (a.x + a.w / 2)
    dy = (b.y + b.h / 2) - (a.y + a.h / 2)
    d = math.hypot(dx, dy)
    p = border_point(a, dx, dy)
    q = border_point(b, -dx, -dy)
    ux, uy = dx / d, dy / d
    return (p[0] + ux * gap, p[1] + uy * gap), (q[0] - ux * gap, q[1] - uy * gap)


def draw_board(board: Board, scale=2) -> Image.Image:
    S = scale
    img = Image.new("RGB", (board.w * S, board.h * S), "white")
    d = ImageDraw.Draw(img)

    def P(*v):
        return [c * S for c in v]

    def draw_text_block(s, x, y, size, bold, anchor_center=False, w=0, h=0):
        f = font(size, bold, S)
        lines = s.split("\n")
        lh = size * LINE_HEIGHT
        top = y + (h - lh * len(lines)) / 2 if anchor_center else y
        for i, line in enumerate(lines):
            ly = top + i * lh + lh / 2
            if anchor_center:
                d.text(P(x + w / 2, ly), line, font=f, fill="#1e1e1e", anchor="mm")
            else:
                d.text(P(x, ly), line, font=f, fill="#1e1e1e", anchor="lm")

    def dashed_rect(x0, y0, x1, y1, color, width, dash=12, gap=8):
        for (ax, ay, bx, by) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
            length = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / length, (by - ay) / length
            t = 0.0
            while t < length:
                t2 = min(t + dash, length)
                d.line(P(ax + ux * t, ay + uy * t, ax + ux * t2, ay + uy * t2), fill=color, width=width)
                t = t2 + gap

    def draw(node):
        if isinstance(node, Text):
            size = node.h / (len(node.text.split("\n")) * LINE_HEIGHT)
            draw_text_block(node.text, node.x, node.y, size, bool(node.style.get("bold")))
            return
        st = node.style
        fill = FILL.get(st.get("fill")) if st.get("fill") else None
        stroke = STROKE.get(st.get("stroke", "black"), "#1e1e1e")
        width = 2 * S
        box = P(node.x, node.y, node.x + node.w, node.y + node.h)
        if node.kind == "ellipse":
            d.ellipse(box, fill=fill, outline=stroke, width=width)
        elif node.kind == "diamond":
            cx, cy = node.x + node.w / 2, node.y + node.h / 2
            pts = P(cx, node.y, node.x + node.w, cy, cx, node.y + node.h, node.x, cy)
            d.polygon(pts, fill=fill, outline=stroke, width=width)
        elif st.get("dashed"):
            if fill:
                d.rectangle(box, fill=fill)
            dashed_rect(node.x, node.y, node.x + node.w, node.y + node.h, stroke, width)
        else:
            d.rectangle(box, fill=fill, outline=stroke, width=width)
        if node.label:
            if node.children:
                draw_text_block(node.label, node.x + 16, node.y + 12, LABEL_SIZE, True)
            else:
                draw_text_block(node.label, node.x, node.y, LABEL_SIZE, False, True, node.w, node.h)
        for c in node.children:
            draw(c)

    for n in board.items:
        draw(n)

    for e in board.edges:
        (x0, y0), (x1, y1) = edge_points(e)
        d.line(P(x0, y0, x1, y1), fill="#1e1e1e", width=2 * S)
        ang = math.atan2(y1 - y0, x1 - x0)
        heads = {"->": [(x1, y1, ang)], "<-": [(x0, y0, ang + math.pi)],
                 "<->": [(x1, y1, ang), (x0, y0, ang + math.pi)]}.get(e.kind, [])
        for hx, hy, a in heads:
            L, spread = 16, 0.45
            pts = [(hx, hy), (hx - L * math.cos(a - spread), hy - L * math.sin(a - spread)),
                   (hx - L * math.cos(a + spread), hy - L * math.sin(a + spread))]
            d.polygon([c * S for p in pts for c in p], fill="#1e1e1e")
        if e.label:
            f = font(18, False, S)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            tw = f.getlength(e.label) / S
            d.rectangle(P(mx - tw / 2 - 6, my - 13, mx + tw / 2 + 6, my + 13), fill="white")
            d.text(P(mx, my), e.label, font=f, fill="#1e1e1e", anchor="mm")

    return img.resize((board.w, board.h), Image.LANCZOS)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "samples" / "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    for name, build in SCENES.items():
        board = build()
        draw_board(board).save(out / f"{name}.png")
        (out / f"{name}.board").write_text(serialize(board))
        print(f"wrote {name}")


if __name__ == "__main__":
    main()

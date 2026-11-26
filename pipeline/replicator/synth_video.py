"""Synthetic screen recordings: scripted scenes that change over time, with a
moving mouse cursor, encoded to mp4 (so they carry real compression noise).

Usage:  uv run python -m replicator.synth_video [outdir]
Writes samples/video/NAME.mp4 plus NAME.timeline.txt (what changes when).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import ImageDraw

from .board import Board
from .synth import DASH_ROWS, KANBAN, draw_board, scene_dashboard, scene_flow, scene_kanban

FPS = 10


def dashboard_live(t: float) -> Board:
    rows = [list(r) for r in DASH_ROWS]
    if t >= 6:
        rows.insert(1, ["#1043", "Robert Fox", "$310.00", "Pending"])
    if t >= 9:
        for r in rows:
            if r[0] == "#1041":
                r[3] = "Shipped"
    return scene_dashboard(revenue="$48,210" if t < 3 else "$49,105", rows=[tuple(r) for r in rows])


DASHBOARD_TIMELINE = """0s   dashboard, static
3s   Revenue $48,210 -> $49,105
6s   new order row #1043 Robert Fox $310.00 Pending inserted at the top (rows shift down)
9s   #1041 status Pending -> Shipped
"""


def kanban_move(t: float) -> Board:
    cols = [(title, color, list(cards)) for title, color, cards in KANBAN]
    floating = None
    if 4 <= t < 5:  # dragging "Dark mode" from In Progress to Done
        cols[1][2].remove("Dark mode")
        f = t - 4
        x0, y0 = 450 + 20, 160 + 90  # its slot in In Progress
        x1, y1 = 860 + 20, 160 + 3 * 90  # next free slot in Done
        floating = ("Dark mode", "blue", x0 + f * (x1 - x0), y0 + f * (y1 - y0) - 30 * math.sin(f * math.pi))
    elif t >= 5:
        cols[1][2].remove("Dark mode")
        cols[2][2].append("Dark mode")
    if t >= 8:
        cols[0][2].append("Write tests")
    return scene_kanban([(a, b, c) for a, b, c in cols], floating)


KANBAN_TIMELINE = """0s   kanban, static
4s   card "Dark mode" dragged from In Progress ...
5s   ... dropped at the bottom of Done
8s   new card "Write tests" added to To Do
"""


def flow_build(t: float) -> Board:
    """A flowchart being drawn step by step (whiteboard-building)."""
    full = scene_flow()
    order = ["Order Flow", "Receive order", "Check stock", "In stock?", "Ship order", "Notify customer",
             "SLA: ship within 24h"]
    appear = {name: 2 * i for i, name in enumerate(order)}  # one new thing every 2 s

    def name(n):
        return getattr(n, "label", None) or getattr(n, "text", None)

    items = [n for n in full.items if t >= appear.get(name(n), 0)]
    present = {id(n) for n in items}
    edges = [e for e in full.edges if id(e.src) in present and id(e.dst) in present]
    return Board(full.w, full.h, items, edges)


FLOW_TIMELINE = """0s   title "Order Flow"
2s   + Receive order
4s   + Check stock, arrow from Receive order
6s   + In stock? diamond, arrow from Check stock
8s   + Ship order, arrow "yes"
10s  + Notify customer, arrow "no"
12s  + note "SLA: ship within 24h"
"""

VIDEOS = {
    "dashboard-live": (dashboard_live, 12, DASHBOARD_TIMELINE),
    "kanban-move": (kanban_move, 11, KANBAN_TIMELINE),
    "flow-build": (flow_build, 14, FLOW_TIMELINE),
}


def cursor_at(t: float, w: int, h: int) -> tuple[float, float]:
    """A wandering mouse path (smooth, never still for long)."""
    return (w * (0.5 + 0.35 * math.sin(0.7 * t) * math.cos(0.23 * t)),
            h * (0.5 + 0.35 * math.sin(0.5 * t + 1.3)))


def draw_cursor(img, x: float, y: float):
    d = ImageDraw.Draw(img)
    pts = [(0, 0), (0, 22), (6, 17), (10, 26), (14, 24), (10, 16), (17, 16)]
    d.polygon([(x + px, y + py) for px, py in pts], fill="black", outline="white")


def render(name: str, outdir: Path) -> Path:
    scene, seconds, timeline = VIDEOS[name]
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(seconds * FPS):
            t = i / FPS
            board = scene(t)
            img = draw_board(board)
            draw_cursor(img, *cursor_at(t, board.w, board.h))
            img.save(Path(tmp) / f"{i:04d}.png")
        mp4 = outdir / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    (outdir / f"{name}.timeline.txt").write_text(timeline)
    return mp4


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "samples" / "video")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    for name in VIDEOS:
        print("wrote", render(name, out))


if __name__ == "__main__":
    main()

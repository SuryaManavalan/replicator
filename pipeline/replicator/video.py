"""Video -> live board stream.

Every `interval` seconds of video:
  1. change gate: if the frame barely differs from the last processed one
     (cursor wiggles, compression noise), skip it — no event.
  2. incremental OCR: re-read only regions that changed; keep the rest.
  3. convert to a board and compare with the previous board:
     KEY (full snapshot) on the first frame or a big change, else Δ lines.

Usage:  uv run python -m replicator.video VIDEO [--interval 3] [--out DIR]
Writes DIR/frames/NNNN.png, DIR/boards/NNNN.board, DIR/events.log, DIR/state.board
(default DIR: samples/video/<name>/).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import Board, scale_board, serialize
from .convert import Word, convert, run_ocr
from .diff import diff


# ---------------------------------------------------------------- change gate

def changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of a downscaled frame that visibly changed."""
    def small(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (3, 3), 0)
    return float((cv2.absdiff(small(a), small(b)) > 20).mean())


# ---------------------------------------------------------------- incremental OCR

def _overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _union(a, b):
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return (x, y, max(a[0] + a[2], b[0] + b[2]) - x, max(a[1] + a[3], b[1] + b[3]) - y)


@dataclass
class IncrementalOCR:
    """Keeps the last frame's OCR; re-reads only regions that changed."""
    full_ratio: float = 0.35  # above this much change, just OCR everything
    prev: np.ndarray | None = None
    words: list[Word] = field(default_factory=list)

    def __call__(self, img: np.ndarray) -> tuple[list[Word], str]:
        H, W = img.shape[:2]
        if self.prev is None:
            return self._full(img)
        d = (cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)) > 40)
        d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop compression specks
        d = cv2.dilate(d.astype(np.uint8), np.ones((9, 9), np.uint8))
        if d.mean() > self.full_ratio:
            return self._full(img)
        n, _, stats, _ = cv2.connectedComponentsWithStats(d)
        pad = 24
        regions = []
        for x, y, w, h, area in stats[1:]:
            if area < 150:
                continue
            regions.append((max(x - pad, 0), max(y - pad, 0), min(w + 2 * pad, W - x + pad), min(h + 2 * pad, H - y + pad)))
        if not regions:
            self.prev = img
            return self.words, "reuse"
        # grow regions to swallow any old word they touch, then merge overlaps
        changed = True
        while changed:
            changed = False
            for i, r in enumerate(regions):
                for w in self.words:
                    wb = (w.x, w.y, w.w, w.h)
                    if _overlaps(r, wb) and _union(r, wb) != r:
                        regions[i] = r = _union(r, wb)
                        changed = True
                for j in range(len(regions) - 1, i, -1):
                    if _overlaps(r, regions[j]):
                        regions[i] = r = _union(r, regions.pop(j))
                        changed = True
        keep = [w for w in self.words if not any(_overlaps(r, (w.x, w.y, w.w, w.h)) for r in regions)]
        new = []
        for x, y, w, h in regions:
            if w >= 16 and h >= 16:
                new += run_ocr(img[y:y + h, x:x + w], x, y, crop=True)
        self.prev, self.words = img, keep + new
        return self.words, f"partial x{len(regions)}"

    def _full(self, img):
        self.prev, self.words = img, run_ocr(img)
        return self.words, "full"


# ---------------------------------------------------------------- runner

def stamp(t: float) -> str:
    return f"@{int(t // 60):02d}:{t % 60:04.1f}" if t % 1 else f"@{int(t // 60):02d}:{int(t % 60):02d}"


def run(video: Path, interval: float, out: Path, gate: float, width: int, confirm: float = 0.5,
        verbose=True) -> dict:
    """Process the video every `interval` seconds. When a frame shows a change,
    a second frame `confirm` seconds later must agree before it's announced:
    a cursor crossing a label glitches one frame, and mid-drag positions
    don't hold, so neither gets announced."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    for sub in ("frames", "boards"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).glob("*"):
            f.unlink()

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        return frame if ok else None

    inc = IncrementalOCR()
    idx = 0

    def process(frame) -> tuple[Board, str, dict]:
        nonlocal idx
        t0 = time.perf_counter()
        words, mode = inc(frame)
        t_ocr = time.perf_counter() - t0
        board = scale_board(convert(frame, words).board, width)
        text = serialize(board)
        idx += 1
        cv2.imwrite(str(out / "frames" / f"{idx:04d}.png"), frame)
        (out / "boards" / f"{idx:04d}.board").write_text(text)
        return board, text, {"frame": idx, "ocr": mode, "ms_ocr": round(t_ocr * 1000),
                             "ms_total": round((time.perf_counter() - t0) * 1000)}

    last_frame: np.ndarray | None = None
    committed: Board | None = None  # the state already announced to consumers
    events: list[str] = []
    steps = []
    t = 0.0
    while t < duration:
        frame = grab(t)
        if frame is None:
            break
        step = {"t": round(t, 2)}
        frac = changed_fraction(frame, last_frame) if last_frame is not None else 1.0
        step["changed"] = round(frac, 4)
        if frac < gate:
            step["action"] = "skip"
        else:
            board, text, info = process(frame)
            step.update(info)
            last_frame = frame
            if committed is None:
                events.append(f"{stamp(t)} KEY\n{text}")
                committed, step["action"] = board, "KEY"
            else:
                now = diff(committed, board)
                if now == []:
                    step["action"] = "no-op"
                else:
                    later = grab(min(t + confirm, duration - 1e-3))
                    if later is None:
                        later = frame
                    board2, text2, info2 = process(later)
                    last_frame = later
                    step["confirm"] = info2
                    after = diff(committed, board2)
                    if now is None or after is None:  # big change
                        if after is None and diff(board, board2) is not None and len(diff(board, board2)) <= 2:
                            events.append(f"{stamp(t)} KEY\n{text2}")
                            committed, step["action"] = board2, "KEY"
                        else:
                            step["action"] = "unsettled"
                    else:
                        held = [l for l in now if l in after]
                        if held:
                            events.append(f"{stamp(t)} Δ\n" + "\n".join(held) + "\n")
                            step["action"] = f"Δ {len(held)}"
                            committed = board2 if set(after) == set(held) else (
                                board if set(now) == set(held) else board2)
                        else:
                            step["action"] = "glitch"  # didn't hold: cursor, animation
            (out / "state.board").write_text(serialize(committed))
        steps.append(step)
        if verbose:
            print(json.dumps(step))
        t += interval
    (out / "events.log").write_text("\n".join(events))
    (out / "steps.json").write_text(json.dumps(steps, indent=1))
    return {"steps": steps, "events": len(events)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between sampled frames")
    ap.add_argument("--gate", type=float, default=0.0003,
                    help="min changed fraction to process a frame (cheap to process small changes, so keep low)")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--confirm", type=float, default=0.5, help="seconds later a change must still hold")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.video.with_suffix("")
    run(args.video, args.interval, out, args.gate, args.width, args.confirm)
    print(f"wrote {out}/events.log")


if __name__ == "__main__":
    main()

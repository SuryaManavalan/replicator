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


def _changed_columns(d: np.ndarray, min_w=150, gap=24):
    """Boxes of vertical bands that changed: x-runs where a column of the change
    mask has enough set pixels, each with the y-extent of its changes."""
    H = d.shape[0]
    cols = d.sum(axis=0) > 0.01 * H
    xs = np.flatnonzero(cols)
    if not len(xs):
        return []
    runs, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x > prev + gap:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    boxes = []
    for x0, x1 in runs:
        if x1 - x0 < min_w:
            continue
        rows = np.flatnonzero(d[:, x0:x1 + 1].any(axis=1))
        boxes.append((int(x0), int(rows[0]), int(x1 - x0 + 1), int(rows[-1] - rows[0] + 1)))
    return boxes


def vertical_shift(a: np.ndarray, b: np.ndarray, box, max_frac=0.6) -> int | None:
    """How far the content of `box` scrolled vertically from gray frame a to b.

    Row-brightness profiles give candidate shifts cheaply; the best one is
    verified pixel by pixel. Returns None if it isn't a clean scroll."""
    x, y, w, h = box
    pa = a[y:y + h, x:x + w].astype(np.float32).mean(axis=1)
    pb = b[y:y + h, x:x + w].astype(np.float32).mean(axis=1)
    best, best_err = None, np.inf
    for dy in range(-int(h * max_frac), int(h * max_frac) + 1):
        if dy == 0:
            continue
        if dy > 0:  # content moved down by dy
            ea, eb = pa[:h - dy], pb[dy:]
        else:
            ea, eb = pa[-dy:], pb[:h + dy]
        if len(ea) < 0.3 * h:
            continue
        err = float(np.abs(ea - eb).mean())
        if err < best_err:
            best, best_err = dy, err
    if best is None:
        return None
    dy = best
    A = a[y:y + h, x:x + w]
    B = b[y:y + h, x:x + w]
    ov_a, ov_b = (A[:h - dy], B[dy:]) if dy > 0 else (A[-dy:], B[:h + dy])
    residual = float((cv2.absdiff(ov_a, ov_b) > 40).mean())
    still = float((cv2.absdiff(A, B) > 40).mean())
    return dy if residual < 0.25 * still and residual < 0.03 else None


@dataclass
class IncrementalOCR:
    """Keeps the last frame's OCR; re-reads only regions that changed.

    Scrolling is recognised: a region whose content shifted vertically keeps
    its words (moved by the shift) and only the newly revealed strip is read."""
    full_ratio: float = 0.35  # above this much (non-scroll) change, just OCR everything
    prev: np.ndarray | None = None
    words: list[Word] = field(default_factory=list)
    scrolls: list = field(default_factory=list)  # [(box, dy)] found on the last frame

    def _changed(self, a_gray, b_gray):
        d = cv2.absdiff(a_gray, b_gray) > 40
        d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # compression specks
        return cv2.dilate(d, np.ones((9, 9), np.uint8))

    def __call__(self, img: np.ndarray) -> tuple[list[Word], str]:
        H, W = img.shape[:2]
        self.scrolls = []
        if self.prev is None:
            return self._full(img)
        cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
        d = self._changed(prev, cur)
        words = list(self.words)

        # 1. scrolled regions: shift the old frame and the old words to match.
        #    Candidates are changed vertical bands (layout columns): the changed
        #    pixels of a scrolling feed are fragmented by the gaps between posts.
        virtuals = [prev.copy() for _ in range(3)]  # shift-1, shift, shift+1
        for x, y, w, h in _changed_columns(d):
            if w * h < 0.04 * W * H or h < 120:
                continue
            dy = vertical_shift(prev, cur, (x, y, w, h))
            if dy is None:
                continue
            self.scrolls.append(((x, y, w, h), dy))
            region = prev[y:y + h, x:x + w]
            for v, s in zip(virtuals, (dy - 1, dy, dy + 1)):
                shifted = np.zeros_like(region)  # the newly revealed strip stays "changed"
                if s > 0:
                    shifted[s:] = region[:h - s]
                elif s < 0:
                    shifted[:h + s] = region[-s:]
                else:
                    shifted[:] = region
                v[y:y + h, x:x + w] = shifted
            moved = []
            for wd in words:
                if x <= wd.cx <= x + w and y <= wd.cy <= y + h:
                    ny = wd.y + dy
                    if y <= ny and ny + wd.h <= y + h:
                        moved.append(Word(wd.text, wd.x, ny, wd.w, wd.h, wd.score, wd.bold))
                else:
                    moved.append(wd)
            words = moved
        if self.scrolls:
            # best of the three alignments per pixel: tolerates sub-pixel scroll
            # and compression noise on moved text edges
            best = np.minimum.reduce([cv2.absdiff(v, cur) for v in virtuals])
            r = cv2.morphologyEx((best > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            d = cv2.dilate(r, np.ones((9, 9), np.uint8))

        if d.mean() > self.full_ratio:
            return self._full(img)
        n, _, stats, _ = cv2.connectedComponentsWithStats(d)
        pad = 24
        regions = []
        for x, y, w, h, area in stats[1:]:
            if area < 150:
                continue
            regions.append((max(x - pad, 0), max(y - pad, 0), min(w + 2 * pad, W - x + pad), min(h + 2 * pad, H - y + pad)))
        tag = "scroll " if self.scrolls else ""
        if not regions:
            self.prev, self.words = img, words
            return self.words, tag + "reuse"
        # grow regions to swallow any old word they touch, then merge overlaps
        changed = True
        while changed:
            changed = False
            for i, r in enumerate(regions):
                for wd in words:
                    wb = (wd.x, wd.y, wd.w, wd.h)
                    if _overlaps(r, wb) and _union(r, wb) != r:
                        regions[i] = r = _union(r, wb)
                        changed = True
                for j in range(len(regions) - 1, i, -1):
                    if _overlaps(r, regions[j]):
                        regions[i] = r = _union(r, regions.pop(j))
                        changed = True
        keep = [wd for wd in words if not any(_overlaps(r, (wd.x, wd.y, wd.w, wd.h)) for r in regions)]
        new = []
        for x, y, w, h in regions:
            if w >= 16 and h >= 16:
                new += run_ocr(img[y:y + h, x:x + w], x, y, crop=True)
        self.prev, self.words = img, keep + new
        return self.words, f"{tag}partial x{len(regions)}"

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
    unsettled = 0  # consecutive changed steps that didn't hold (scrolling, playing video)
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
                    settle = diff(board, board2)
                    if now is None or after is None:  # big change
                        if after is None and settle is not None and len(settle) <= 2:
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
                    # A screen that keeps changing (scrolling, a playing video) never
                    # holds still: after two unsettled steps, announce what's there now.
                    unsettled = unsettled + 1 if step["action"] in ("unsettled", "glitch") else 0
                    if unsettled >= 2:
                        latest = diff(committed, board2)
                        if latest is None:
                            events.append(f"{stamp(t)} KEY\n{text2}")
                            step["action"] = "KEY (moving)"
                        elif latest:
                            events.append(f"{stamp(t)} Δ\n" + "\n".join(latest) + "\n")
                            step["action"] = f"Δ {len(latest)} (moving)"
                        committed, unsettled = board2, 0
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

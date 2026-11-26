"""Screenshot -> Board (v0).

Pipeline:
  1. OCR (RapidOCR, PP-OCRv5 English) -> text lines with boxes
  2. Erase text, then edge-detect what's left
  3. Shapes: closed contours classified as rect / ellipse / diamond
  4. Regions: recursive XY-cut along long divider lines (UI panes)
  5. Connectors: straight ink segments joining two shapes (+ arrowheads, labels)
  6. Tree: everything placed inside the smallest thing that contains it
  7. Prune: keep a container only if it separates text
  8. Text: merge wrapped lines into paragraphs; centered single text -> label

Usage:  uv run python -m replicator.convert IMAGE... [--debug]
Writes IMAGE.pred.board next to each image (and a debug overlay with --debug).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import Board, Edge, Shape, Table, Text, _gaps, _split, scale_board, serialize

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"

# Named fills (same palette the viewer uses), RGB.
FILLS = {
    "gray": (233, 236, 239), "red": (255, 201, 201), "pink": (252, 194, 215), "violet": (208, 191, 255),
    "blue": (165, 216, 255), "cyan": (153, 233, 242), "teal": (150, 242, 215), "green": (178, 242, 187),
    "yellow": (255, 236, 153), "orange": (255, 216, 168),
}


# ---------------------------------------------------------------- OCR

@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    score: float
    bold: bool = False

    @property
    def x2(self): return self.x + self.w
    @property
    def y2(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2


DEBUG_CONN = False
_engine = None


def run_ocr(img: np.ndarray, dx: int = 0, dy: int = 0, crop: bool = False) -> list[Word]:
    """OCR an image (or a crop of one, offset by dx, dy). No caching.
    crop=True uses an engine that never upscales small inputs: the default
    upscales to a 736px minimum side, which is slow and splits words differently."""
    r = (_get_crop_engine() if crop else _get_engine())(img)
    words = []
    if r.boxes is not None:
        for box, txt, score in zip(r.boxes, r.txts, r.scores):
            xs, ys = box[:, 0], box[:, 1]
            words.append(Word(txt.strip(), int(xs.min()) + dx, int(ys.min()) + dy,
                              int(xs.max() - xs.min()), int(ys.max() - ys.min()), float(score)))
    return words


def ocr(img: np.ndarray) -> list[Word]:
    """Whole-image OCR, cached on disk by image hash (for stills)."""
    key = hashlib.sha1(img.tobytes()).hexdigest()[:16]
    cached = CACHE / f"ocr-{key}.json"
    if cached.exists():
        return [Word(**d) for d in json.loads(cached.read_text())]
    words = run_ocr(img)
    CACHE.mkdir(exist_ok=True)
    cached.write_text(json.dumps([w.__dict__ for w in words]))
    return words


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        _engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,  # screens are never rotated; saves ~0.5 s
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.EN,
        })
    return _engine


def stroke_ratio(gray: np.ndarray, w: Word) -> float | None:
    """Stroke width relative to text height: 2*area/perimeter of the glyph ink."""
    patch = gray[w.y:w.y2, w.x:w.x2]
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    t, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (patch < t) if (patch >= t).mean() > 0.5 else (patch >= t)  # glyphs are the minority
    ink = ink.astype(np.uint8)
    area = int(ink.sum())
    if area < 20:
        return None
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    return (2 * area / max(perim, 1)) / w.h if perim else None


def mark_bold(gray: np.ndarray, words: list[Word]) -> None:
    """Bold = noticeably thicker strokes than the screen's typical text."""
    ratios = [stroke_ratio(gray, w) for w in words]
    vals = [r for r in ratios if r is not None]
    if len(vals) < 4:
        return
    med = float(np.median(vals))
    for w, r in zip(words, ratios):
        w.bold = r is not None and r > 1.3 * med


_crop_engine = None


def _get_crop_engine():
    global _crop_engine
    if _crop_engine is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        _crop_engine = RapidOCR(params={
            "Global.log_level": "error", "Global.use_cls": False,
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
            "Det.limit_type": "max", "Det.limit_side_len": 2000,  # keep the crop at native scale
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.EN,
        })
    return _crop_engine


def all_words(words: list[Word]) -> list[Word]:
    """Every plausible OCR box, including ones too unsure to keep as text: used
    for erasing glyphs so they don't come back as icons or shapes."""
    return [w for w in words if w.score >= 0.3]


def ocr_inside(img: np.ndarray, shapes: list, words: list[Word], med_h: float):
    """Detection misses lone glyphs (a "1" in a circle); recognise small textless
    shapes' interiors directly, in one batched recognition call.

    Returns (words found, shapes they were found in)."""
    from rapidocr.ch_ppocr_rec.typings import TextRecInput

    cands, crops, boxes = [], [], []
    for n in sorted(shapes, key=lambda n: -n.area):  # outer shapes first
        if n.kind not in ("rect", "ellipse", "diamond") or max(n.w, n.h) > 4 * med_h:
            continue
        if n.kind == "rect" and not 0.6 < n.w / n.h < 1.7:
            continue  # lone glyphs live in circles, diamonds and squarish boxes
        if any(n.contains(w, 0) or (n.x <= w.cx <= n.x2 and n.y <= w.cy <= n.y2) for w in words):
            continue
        if any(c.contains(n, 0) for c in cands):
            continue  # a glyph hole inside a candidate
        ix, iy = int(n.w * 0.18), int(n.h * 0.18)  # stay clear of the outline
        crop = img[n.y + iy:n.y2 - iy, n.x + ix:n.x2 - ix]
        if crop.size == 0:
            continue
        cands.append(n)
        crops.append(crop)
        boxes.append((n.x + ix, n.y + iy, n.w - 2 * ix, n.h - 2 * iy))
    if not crops:
        return [], []
    out = _get_engine()._load_rec_model()(TextRecInput(img=crops))
    found, filled = [], []
    for n, box, txt, score in zip(cands, boxes, out.txts, out.scores):
        txt = txt.strip()
        if score >= 0.8 and any(ch.isalnum() for ch in txt):
            found.append(Word(txt, *box, float(score)))
            filled.append(n)
    return found, filled


def clean_words(words: list[Word]) -> list[Word]:
    for w in words:
        w.text = w.text.strip().strip("→←↑↓⟶⟵»«·•").strip()
    words = [w for w in words if w.text and w.score >= 0.6]
    if not words:
        return words
    med_h = float(np.median([w.h for w in words]))
    out = []
    for w in words:
        if len(w.text) <= 2 and w.score < 0.85:
            continue  # icons read as a character or two
        if len(w.text) == 1 and (w.h < 0.6 * med_h or w.score < 0.9 or w.text in "|Il!.,'-_~"):
            continue  # specks on lines/icons read as a character
        out.append(w)
    return out


# ---------------------------------------------------------------- image prep

def erase_text(img: np.ndarray, words: list[Word], pad=3) -> np.ndarray:
    """Erase glyphs inside each text box, painting them with the box's background.

    Only ink blobs entirely inside the (padded) box are erased; anything that
    crosses the box edge is a line or border passing through and is kept.
    """
    out = img.copy()
    H, W = img.shape[:2]
    for w in words:
        x0, y0 = max(w.x - pad, 0), max(w.y - pad, 0)
        x1, y1 = min(w.x2 + pad, W), min(w.y2 + pad, H)
        box = img[y0:y1, x0:x1]
        if box.size == 0:
            continue
        bg = np.median(box.reshape(-1, 3), axis=0)
        ink = (np.abs(box.astype(np.int16) - bg).max(axis=2) > 40).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        bh, bw = ink.shape
        erase = np.zeros_like(ink, bool)
        for i in range(1, n):
            cx, cy, cw, ch = stats[i, :4]
            touches_edge = not (cx > 0 and cy > 0 and cx + cw < bw and cy + ch < bh)
            # horizontal line piece (arrow stub) passing through; vertical stems are glyphs (l, I)
            thin_stroke = ch <= 3 and cw >= max(10, 0.6 * w.h)
            if not touches_edge and not thin_stroke:
                erase |= labels == i
        # anti-aliased glyph edges sit just outside the thresholded blob
        erase = cv2.dilate(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[y0:y1, x0:x1][erase] = bg
    return out


def bridge_lines(img: np.ndarray, words: list[Word], pad=3, reach=24) -> np.ndarray:
    """Redraw thin lines that a text label interrupts (a label sitting on an
    arrow or on a box border). A column/row counts as a line passing through
    when ink continues on BOTH sides of the box and is thin (not a border
    running along the box)."""
    out = img.copy()
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    for w in words:
        x0, y0 = max(w.x - pad, 0), max(w.y - pad, 0)
        x1, y1 = min(w.x2 + pad, W - 1), min(w.y2 + pad, H - 1)
        if y0 - reach < 0 or y1 + reach >= H or x0 - reach < 0 or x1 + reach >= W:
            continue
        bg = float(np.median(gray[y0:y1, x0:x1]))
        def is_ink(region):
            return np.abs(region - bg) > 40

        def runs_out(strip: np.ndarray) -> np.ndarray:
            """Per column of `strip` (row 0 = next to the box): does ink start
            within reach and continue outward for >= 5 px? A line passing
            through does; a border it merely bumps into doesn't."""
            first = np.argmax(strip, axis=0)
            hit = strip.any(axis=0)
            ok = np.zeros(strip.shape[1], bool)
            for c in np.flatnonzero(hit):
                k = first[c]
                ok[c] = strip[k:k + 5, c].sum() >= 5
            return ok

        # vertical lines crossing top and bottom edges
        up = is_ink(gray[y0 - reach:y0, x0:x1])[::-1]  # row 0 = just above the box
        down = is_ink(gray[y1:y1 + reach, x0:x1])
        cols = np.flatnonzero(runs_out(up) & runs_out(down))
        if 0 < len(cols) <= 6:  # a thin line, not a border running along the box
            for c in cols:
                ya = y0 - 1 - int(np.argmax(up[:, c]))
                yb = y1 + int(np.argmax(down[:, c]))
                colour = img[ya, x0 + int(c)].tolist()
                cv2.line(out, (x0 + int(c), ya), (x0 + int(c), yb), colour, 1)
        # horizontal lines crossing left and right edges
        lt = is_ink(gray[y0:y1, x0 - reach:x0])[:, ::-1]  # col 0 = just left of the box
        rt = is_ink(gray[y0:y1, x1:x1 + reach])
        rows = np.flatnonzero(runs_out(lt.T) & runs_out(rt.T))
        if 0 < len(rows) <= 6:
            for r in rows:
                xa = x0 - 1 - int(np.argmax(lt[r]))
                xb = x1 + int(np.argmax(rt[r]))
                colour = img[y0 + int(r), xa].tolist()
                cv2.line(out, (xa, y0 + int(r)), (xb, y0 + int(r)), colour, 1)
    return out


def edge_map(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    e = np.zeros(img.shape[:2], np.uint8)
    for c in cv2.split(blur):
        e |= cv2.Canny(c, 8, 24)
    return e


# ---------------------------------------------------------------- shapes

@dataclass
class Node:
    kind: str  # rect | ellipse | diamond
    x: int
    y: int
    w: int
    h: int
    implicit: bool = False  # region from XY-cut rather than a drawn shape
    words: list[Word] = field(default_factory=list)
    kids: list[Node] = field(default_factory=list)
    connected: bool = False
    fill: str | None = None
    shape: Shape | None = None

    @property
    def area(self): return self.w * self.h
    @property
    def x2(self): return self.x + self.w
    @property
    def y2(self): return self.y + self.h

    def contains(self, o, tol=6) -> bool:
        return (o.x >= self.x - tol and o.y >= self.y - tol and
                o.x2 <= self.x2 + tol and o.y2 <= self.y2 + tol)


def iou(a, b) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    inter = ix * iy
    return inter / (a.area + b.area - inter) if inter else 0.0


def _is_ellipse(c, w, h, ratio) -> bool:
    """Convex hull fills ~pi/4 of the box. Using the hull tolerates notches where
    lines, arrowheads or leftover glyph pixels touch the outline from inside."""
    hull = cv2.contourArea(cv2.convexHull(c)) / (w * h)
    return 0.72 < hull < 0.84 and ratio / hull > 0.72


def _rect_outline_support(edges: np.ndarray, x, y, w, h, step=3) -> float:
    """Fraction of the bbox perimeter lying on edge pixels."""
    H, W = edges.shape
    xs = np.arange(x, x + w, step).clip(0, W - 1)
    ys = np.arange(y, y + h, step).clip(0, H - 1)
    y2, x2 = min(y + h - 1, H - 1), min(x + w - 1, W - 1)
    vals = np.concatenate([edges[y, xs], edges[y2, xs], edges[ys, x], edges[ys, x2]])
    return float((vals > 0).mean())


def _outline_support(edges: np.ndarray, x, y, w, h) -> float:
    """Fraction of the bbox-inscribed ellipse outline that lies on edge pixels."""
    pts = cv2.ellipse2Poly((x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 4)
    H, W = edges.shape
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
    if not len(pts):
        return 0.0
    return float((edges[pts[:, 1], pts[:, 0]] > 0).mean())


def _is_diamond(c, x, y, w, h) -> bool:
    """Extreme points sit near the middles of the bounding box sides."""
    pts = c.reshape(-1, 2)
    top = pts[pts[:, 1].argmin()]; bot = pts[pts[:, 1].argmax()]
    left = pts[pts[:, 0].argmin()]; right = pts[pts[:, 0].argmax()]
    cx, cy = x + w / 2, y + h / 2
    return (abs(top[0] - cx) < 0.15 * w and abs(bot[0] - cx) < 0.15 * w
            and abs(left[1] - cy) < 0.15 * h and abs(right[1] - cy) < 0.15 * h)


def find_shapes(edges: np.ndarray, min_size=24) -> list[Node]:
    H, W = edges.shape
    plain = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    # a second pass bridges gaps along straight lines (dashed borders); it also
    # blunts sharp corners, so diamonds/ellipses come from the plain pass.
    bridged = plain | cv2.morphologyEx(plain, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    bridged |= cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    near = cv2.dilate(plain, np.ones((5, 5), np.uint8))  # outline support tolerance
    contours = []
    for m in (plain, bridged):
        contours += cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cands: list[Node] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_size or h < min_size or (w > W - 8 and h > H - 8):
            continue
        ratio = cv2.contourArea(c) / (w * h)
        poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if ratio > 0.88:
            kind = "rect"
        elif w > 3 * min_size and h > 3 * min_size and _rect_outline_support(near, x, y, w, h) > 0.92 \
                and cv2.contourArea(cv2.convexHull(c)) / (w * h) > 0.95:
            kind = "rect"  # a container whose inside is cut into by attached lines/boxes
        elif 0.4 < w / h < 2.5 and len(poly) > 6 and _is_ellipse(c, w, h, ratio) \
                and _outline_support(near, x, y, w, h) > 0.8:
            kind = "ellipse"
        elif 0.42 < ratio < 0.6 and 4 <= len(poly) <= 6 and _is_diamond(c, x, y, w, h):
            kind = "diamond"
        else:
            continue
        cands.append(Node(kind, x, y, w, h))
    # A stroked outline yields an outer and an inner contour: keep the outer.
    cands.sort(key=lambda n: -n.area)
    kept: list[Node] = []
    for n in cands:
        if all(iou(n, k) < 0.85 and not _same_outline(k, n) for k in kept):
            kept.append(n)
    return kept


def _same_outline(outer: Node, inner: Node, margin=10) -> bool:
    """Inner/outer contours of one stroked outline."""
    return (0 <= inner.x - outer.x <= margin and 0 <= inner.y - outer.y <= margin
            and 0 <= outer.x2 - inner.x2 <= margin and 0 <= outer.y2 - inner.y2 <= margin)


def find_regions(edges: np.ndarray, shapes: list[Node]) -> list[Node]:
    """Recursive XY-cut along straight dividers that span (almost) a whole region."""
    H, W = edges.shape
    out: list[Node] = []
    edges = edges.copy()
    for s in shapes:  # a shape's own border is not a divider
        cv2.rectangle(edges, (s.x, s.y), (s.x2, s.y2), 0, 9)

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        groups, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i > prev + 2:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        return groups

    def cut(x0, y0, x1, y1, depth):
        if depth > 6 or x1 - x0 < 80 or y1 - y0 < 40:
            return
        sub = edges[y0:y1, x0:x1] > 0
        rows = [(a, b) for a, b in runs(sub.mean(axis=1) > 0.85) if 8 < a and b < (y1 - y0) - 8]
        cols = [(a, b) for a, b in runs(sub.mean(axis=0) > 0.85) if 8 < a and b < (x1 - x0) - 8]
        if not rows and not cols:
            return
        vertical = bool(cols) and (not rows or len(cols) >= len(rows))
        cuts = cols if vertical else rows
        bounds = [0] + [v for a, b in cuts for v in (a, b)] + [(x1 - x0) if vertical else (y1 - y0)]
        for i in range(0, len(bounds), 2):
            a, b = bounds[i], bounds[i + 1]
            if b - a < 30:
                continue
            r = Node("rect", x0 + a, y0, b - a, y1 - y0, implicit=True) if vertical \
                else Node("rect", x0, y0 + a, x1 - x0, b - a, implicit=True)
            out.append(r)
            cut(r.x, r.y, r.x2, r.y2, depth + 1)

    cut(0, 0, W, H, 0)
    return [r for r in out if all(iou(r, s) < 0.85 for s in shapes)]


def shape_fill(img: np.ndarray, n: Node, bg: np.ndarray) -> str | None:
    inset = 5
    patch = img[n.y + inset:n.y2 - inset, n.x + inset:n.x2 - inset]
    if patch.size == 0:
        return None
    if n.kind != "rect":  # sample the middle band only
        h, w = patch.shape[:2]
        patch = patch[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
    b, g, r = np.median(patch.reshape(-1, 3), axis=0)
    rgb = np.array([r, g, b], float)
    if np.abs(rgb - bg).max() < 12:
        return None
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    sat, val = hsv[1] / 255, hsv[2] / 255
    if val < 0.5:
        return None  # dark fills (dark-mode UIs) don't translate to a light board
    if sat < 0.15:
        return "gray"
    name, dist = min(((k, np.linalg.norm(rgb - np.array(v))) for k, v in FILLS.items() if k != "gray"),
                     key=lambda t: t[1])
    return name if dist < 90 else "#%02x%02x%02x" % (int(r), int(g), int(b))


# ---------------------------------------------------------------- images + icons

def is_photo(img: np.ndarray, n: Node) -> bool:
    """Photos/illustrations: many distinct colours and lots of tonal variation."""
    inset = 3
    patch = img[n.y + inset:n.y2 - inset, n.x + inset:n.x2 - inset]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        return False
    step = max(1, int(math.sqrt(patch.shape[0] * patch.shape[1] / 40000)))
    q = (patch[::step, ::step] >> 4).reshape(-1, 3).astype(np.int32)  # 16 levels per channel
    counts = np.bincount(q[:, 0] << 8 | q[:, 1] << 4 | q[:, 2], minlength=4096)
    dominant = counts.max() / counts.sum()  # UI panels are mostly one background colour
    return np.count_nonzero(counts) > 120 and float(q.std()) > 1.2 and dominant < 0.35


def find_icons(clean: np.ndarray, words: list[Word], shapes: list[Node], conns, med_h: float) -> list[Node]:
    """Small compact ink blobs that aren't text, lines or shape outlines."""
    ink = _ink(clean)
    for n in shapes:
        if n.kind == "image":
            cv2.rectangle(ink, (n.x - 2, n.y - 2), (n.x2 + 2, n.y2 + 2), 0, -1)
        elif n.kind == "ellipse":
            cv2.ellipse(ink, (n.x + n.w // 2, n.y + n.h // 2), (n.w // 2, n.h // 2), 0, 0, 360, 0, 7)
        else:
            cv2.rectangle(ink, (n.x, n.y), (n.x2, n.y2), 0, 7)
            r = int(min(n.w, n.h, 48) / 2)  # rounded corners curve inside the box
            for cx, cy in ((n.x, n.y), (n.x2 - r, n.y), (n.x, n.y2 - r), (n.x2 - r, n.y2 - r)):
                ink[cy:cy + r, cx:cx + r] = 0
    for *_, sg in conns:
        cv2.line(ink, (int(sg.x0), int(sg.y0)), (int(sg.x1), int(sg.y1)), 0, 9)
    joined = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    lo, hi = 0.35 * med_h, 2.4 * med_h
    icons = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        x, y, w, h = x + 2, y + 2, w - 4, h - 4  # undo the dilation
        if not (lo <= w <= hi and lo <= h <= hi and 0.4 < w / max(h, 1) < 2.5):
            continue
        if np.count_nonzero(ink[y:y + h, x:x + w]) < 0.06 * w * h:
            continue
        cx, cy = x + w / 2, y + h / 2
        if any(wd.x <= cx <= wd.x2 and wd.y <= cy <= wd.y2 for wd in words):
            continue  # leftover text pixels
        icons.append(Node("icon", int(x), int(y), int(w), int(h)))
    return icons


def misread_icons(words: list[Word], shapes: list[Node]) -> tuple[list[Word], list[Node]]:
    """A 1-2 character 'word' just left of a real label is usually its icon."""
    keep, icons = [], []
    for w in words:
        short = len(w.text.strip()) <= 3 and w.w < 1.6 * w.h  # icons are squarish
        if short:
            sole_label = any(s.contains(w, 2) and s.w < 4 * w.w + 40 for s in shapes)
            label_right = any(
                o is not w and len(o.text) >= 3 and o.score >= 0.6 and abs(o.cy - w.cy) < 0.4 * max(o.h, w.h)
                and -0.5 * w.h <= o.x - w.x2 < 2 * w.h for o in words)
            if label_right and not sole_label:
                icons.append(Node("icon", w.x, w.y, w.w, w.h))
                continue
        keep.append(w)
    return keep, icons


# ---------------------------------------------------------------- tables

def find_tables(words: list[Word]) -> tuple[list[Table], list[Word]]:
    """Runs of >=3 rows whose cells line up in >=2 shared columns."""
    if len(words) < 6:
        return [], words
    med_h = float(np.median([w.h for w in words]))
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows and abs(rows[-1][0].cy - w.cy) < 0.4 * med_h:
            rows[-1].append(w)
        else:
            rows.append([w])
    for r in rows:
        r.sort(key=lambda w: w.x)

    tables, used = [], set()
    i = 0
    while i < len(rows):
        if len(rows[i]) < 2:
            i += 1
            continue
        cols = [w.x for w in rows[i]]
        tol = max(10, 0.8 * med_h)
        run = [rows[i]]
        pitch = None
        for r in rows[i + 1:]:
            if len(r) < 2 or len(r) > len(cols) + 1:
                break
            dy = r[0].cy - run[-1][0].cy
            if pitch is not None and abs(dy - pitch) > 0.35 * pitch:
                break
            if not all(any(abs(w.x - c) < tol for c in cols) for w in r):
                break
            pitch = dy
            run.append(r)
        if len(run) >= 3:
            ws = [w for r in run for w in r]
            x, y = min(w.x for w in ws), min(w.y for w in ws)
            grid = [["" for _ in cols] for _ in run]
            for ri, r in enumerate(run):
                for w in r:
                    ci = min(range(len(cols)), key=lambda c: abs(cols[c] - w.x))
                    grid[ri][ci] = (grid[ri][ci] + " " + w.text).strip()
            size = round(float(np.median([w.h for w in ws])) / 1.25)
            tables.append(Table(x, y, max(w.x2 for w in ws) - x, max(w.y2 for w in ws) - y,
                                cols, grid, {"size": size}))
            used.update(id(w) for w in ws)
            i += len(run)
        else:
            i += 1
    return tables, [w for w in words if id(w) not in used]


# ---------------------------------------------------------------- connectors

@dataclass
class Seg:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def length(self): return math.hypot(self.x1 - self.x0, self.y1 - self.y0)
    @property
    def angle(self): return math.atan2(self.y1 - self.y0, self.x1 - self.x0) % math.pi


def _point_seg_dist(px, py, s: Seg) -> tuple[float, float]:
    """(perpendicular distance, projection t in [0,1] units of length)."""
    dx, dy = s.x1 - s.x0, s.y1 - s.y0
    L2 = dx * dx + dy * dy or 1
    t = ((px - s.x0) * dx + (py - s.y0) * dy) / L2
    qx, qy = s.x0 + t * dx, s.y0 + t * dy
    return math.hypot(px - qx, py - qy), t


def _inside_any(px, py, shapes, shrink=3) -> bool:
    return any(n.x + shrink < px < n.x2 - shrink and n.y + shrink < py < n.y2 - shrink for n in shapes)


def merge_segments(segs: list[Seg], words: list[Word], shapes: list) -> list[Seg]:
    """Join collinear pieces into whole lines.

    Pieces are grouped by direction and perpendicular offset (Hough normal
    form), then merged along the line in one sorted pass. Small gaps are
    bridged; big ones only where a text label interrupts the line, and never
    through a shape.
    """
    if not segs:
        return []
    ang_tol, off_tol = math.radians(4), 5.0
    info = []
    for s in segs:
        th = s.angle  # [0, pi)
        nx, ny = -math.sin(th), math.cos(th)
        info.append((th, s.x0 * nx + s.y0 * ny, s))
    info.sort(key=lambda t: t[0])

    # cluster: consecutive in angle, then split by offset
    groups: list[list] = []
    for t in info:
        if groups and t[0] - groups[-1][-1][0] <= ang_tol:
            groups[-1].append(t)
        else:
            groups.append([t])
    if len(groups) > 1 and math.pi - info[-1][0] + info[0][0] <= ang_tol:
        groups[0] = groups.pop() + groups[0]  # wrap around near 0/180 degrees

    out: list[Seg] = []
    for g in groups:
        # angle near 0 and pi flip the normal's sign: normalise each piece's offset to the group's first direction
        th0 = g[0][0]
        nx, ny = -math.sin(th0), math.cos(th0)
        dx, dy = math.cos(th0), math.sin(th0)
        g = sorted(((s.x0 * nx + s.y0 * ny, s) for _, _, s in g), key=lambda t: t[0])
        lines: list[list] = []
        for off, s in g:
            if lines and off - lines[-1][-1][0] <= off_tol:
                lines[-1].append((off, s))
            else:
                lines.append([(off, s)])
        for line in lines:
            off = float(np.median([o for o, _ in line]))
            spans = sorted((min(a_, b_), max(a_, b_)) for _, s in line
                           for a_, b_ in [((s.x0 * dx + s.y0 * dy), (s.x1 * dx + s.y1 * dy))])
            cur = list(spans[0])
            merged = []
            for lo, hi in spans[1:]:
                gap = lo - cur[1]
                ok = gap <= 12
                if not ok or gap > 0:
                    pts = [(t * dx + off * nx, t * dy + off * ny) for t in np.linspace(cur[1], lo, 7)]
                    through_shape = any(_inside_any(px, py, shapes, shrink=8) for px, py in pts[1:-1])
                    if not ok and not through_shape:
                        mx, my = pts[3]
                        ok = any(w.x - 10 <= mx <= w.x2 + 10 and w.y - 10 <= my <= w.y2 + 10 for w in words)
                    elif through_shape:
                        ok = False
                if ok:
                    cur[1] = max(cur[1], hi)
                else:
                    merged.append(cur)
                    cur = [lo, hi]
            merged.append(cur)
            for lo, hi in merged:
                out.append(Seg(lo * dx + off * nx, lo * dy + off * ny, hi * dx + off * nx, hi * dy + off * ny))
    return out


def _dist_to_node(px, py, n: Node) -> float:
    if n.kind == "ellipse":
        cx, cy, a, b = n.x + n.w / 2, n.y + n.h / 2, n.w / 2, n.h / 2
        r = math.hypot((px - cx) / a, (py - cy) / b)
        return abs(r - 1) * min(a, b)
    dx = max(n.x - px, 0, px - n.x2)
    dy = max(n.y - py, 0, py - n.y2)
    if dx == 0 and dy == 0:  # inside: distance to border
        return min(px - n.x, n.x2 - px, py - n.y, n.y2 - py)
    return math.hypot(dx, dy)


def _ink(img: np.ndarray) -> np.ndarray:
    """Pixels that stand out from their local background (either polarity)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    return (np.abs(gray.astype(np.int16) - bg.astype(np.int16)) > 40).astype(np.uint8) * 255


def _mask_shapes(ink: np.ndarray, shapes: list[Node], grow: int) -> np.ndarray:
    """Blank out shapes. Containers (shapes with other shapes inside) only lose
    their border band, so arrows drawn inside them survive."""
    ink = ink.copy()
    for n in shapes:
        container = any(o is not n and n.contains(o, 0) for o in shapes)
        thick = -1 if not container else 2 * grow + 3
        if n.kind == "ellipse":
            cv2.ellipse(ink, (n.x + n.w // 2, n.y + n.h // 2), (n.w // 2 + grow, n.h // 2 + grow), 0, 0, 360, 0, thick)
        else:
            cv2.rectangle(ink, (n.x - grow, n.y - grow), (n.x2 + grow, n.y2 + grow), 0, thick)
    return ink


def _border_point(n: Node, dx: float, dy: float) -> tuple[float, float]:
    cx, cy, hw, hh = n.x + n.w / 2, n.y + n.h / 2, n.w / 2, n.h / 2
    if n.kind == "ellipse":
        t = 1 / math.hypot(dx / hw, dy / hh)
    elif n.kind == "diamond":
        t = 1 / (abs(dx) / hw + abs(dy) / hh)
    else:
        t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


def find_connectors(img: np.ndarray, clean: np.ndarray, shapes: list[Node], words: list[Word]):
    H, W = img.shape[:2]
    raw_ink = _ink(img)
    head_ink = _mask_shapes(raw_ink, shapes, 2)
    line_ink = cv2.dilate(raw_ink, np.ones((5, 5), np.uint8))
    text_mask = np.zeros((H, W), bool)
    for w in words:
        text_mask[max(w.y - 2, 0):w.y2 + 2, max(w.x - 2, 0):w.x2 + 2] = True

    def has_head(tx, ty, ox, oy) -> bool:
        """Arrowhead at (tx,ty) on a line coming from (ox,oy)?"""
        L = math.hypot(ox - tx, oy - ty) or 1
        ux, uy = (ox - tx) / L, (oy - ty) / L
        r = 24
        x0, y0 = int(max(tx - r, 0)), int(max(ty - r, 0))
        ys, xs = np.nonzero(head_ink[y0:int(ty + r), x0:int(tx + r)])
        px, py = xs + x0 - tx, ys + y0 - ty
        along = px * ux + py * uy
        perp = px * uy - py * ux
        near = (along > -2) & (along < 24)
        cone = np.abs(perp) < 0.6 * along + 2  # a head widens away from its tip
        left = int((near & cone & (perp > 2.5)).sum())
        right = int((near & cone & (perp < -2.5)).sum())
        return min(left, right) >= 6  # an arrowhead flares out on both sides

    found: dict[frozenset, tuple] = {}

    def add(a: Node, b: Node, s: Seg, source: str):
        key = frozenset((id(a), id(b)))
        if key in found and found[key][4] == "direct":
            return
        ha, hb = has_head(s.x0, s.y0, s.x1, s.y1), has_head(s.x1, s.y1, s.x0, s.y0)
        if ha and not hb:
            a, b, ha, hb, s = b, a, hb, ha, Seg(s.x1, s.y1, s.x0, s.y0)
        kind = "<->" if ha and hb else "->" if hb else "--"
        found[key] = (a, b, kind, s, source)

    # 1. Direct test: is the straight path between two shapes drawn in ink?
    #    Text boxes are "don't care", so labels sitting on the line don't break it.
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            if a.contains(b) or b.contains(a):
                continue
            dx = (b.x + b.w / 2) - (a.x + a.w / 2)
            dy = (b.y + b.h / 2) - (a.y + a.h / 2)
            if not dx and not dy:
                continue
            p = _border_point(a, dx, dy)
            q = _border_point(b, -dx, -dy)
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            if L < 8 or (q[0] - p[0]) * dx + (q[1] - p[1]) * dy < 0:
                continue  # touching or overlapping
            ts = np.linspace(0, 1, max(int(L / 2), 4))[1:-1]
            xs = (p[0] + ts * (q[0] - p[0])).astype(int).clip(0, W - 1)
            ys = (p[1] + ts * (q[1] - p[1])).astype(int).clip(0, H - 1)
            vis = ~text_mask[ys, xs]
            if vis.sum() < 4 or (line_ink[ys, xs][vis] > 0).mean() < 0.85:
                continue
            # only now (rarely reached) check the path doesn't run through a third shape
            others = [n for n in shapes if n is not a and n is not b and not n.contains(a) and not n.contains(b)]
            if any(_inside_any(x, y, others, shrink=8) for x, y in zip(xs[::4], ys[::4])):
                continue
            add(a, b, Seg(*p, *q), "direct")

    # 2. Hough segments (lines that don't run centre-to-centre)
    ink = _mask_shapes(_ink(clean), shapes, 4)
    lines = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=10, minLineLength=10, maxLineGap=10)
    segs = [Seg(*map(float, l)) for l in lines.reshape(-1, 4)] if lines is not None else []
    segs = merge_segments(segs, words, shapes)

    def attach(px, py):
        best = min(shapes, key=lambda n: _dist_to_node(px, py, n), default=None)
        return best if best and _dist_to_node(px, py, best) < 24 else None

    direct = [f[3] for f in found.values() if f[4] == "direct"]

    def along_direct(s: Seg) -> bool:
        """A piece of a line we already have (e.g. split where it grazes a box)."""
        for d in direct:
            da = abs(s.angle - d.angle)
            if min(da, math.pi - da) > math.radians(4):
                continue
            (d0, t0), (d1, t1) = _point_seg_dist(s.x0, s.y0, d), _point_seg_dist(s.x1, s.y1, d)
            if max(d0, d1) < 10 and -0.05 <= min(t0, t1) and max(t0, t1) <= 1.05:
                return True
        return False

    def isolated(s: Seg) -> bool:
        """Connectors run through empty space; strokes of unread text (other
        scripts, logos) have ink right beside them."""
        L = s.length or 1
        nx, ny = -(s.y1 - s.y0) / L, (s.x1 - s.x0) / L
        ts = np.linspace(0.15, 0.85, 15)
        busy = 0
        for t in ts:
            px, py = s.x0 + t * (s.x1 - s.x0), s.y0 + t * (s.y1 - s.y0)
            for off in (-9, -6, 6, 9):
                qx, qy = int(px + off * nx), int(py + off * ny)
                if 0 <= qx < W and 0 <= qy < H and raw_ink[qy, qx]:
                    busy += 1
                    break
        return busy / len(ts) < 0.3

    def inked(s: Seg) -> bool:
        """Most of the merged line (outside text boxes) must actually be drawn."""
        n = max(int(s.length / 2), 4)
        ts = np.linspace(0, 1, n)
        xs = (s.x0 + ts * (s.x1 - s.x0)).astype(int).clip(0, W - 1)
        ys = (s.y0 + ts * (s.y1 - s.y0)).astype(int).clip(0, H - 1)
        vis = ~text_mask[ys, xs]
        return vis.sum() >= 4 and (line_ink[ys, xs][vis] > 0).mean() >= 0.75

    for s in segs:
        if s.length < 20 or along_direct(s) or not isolated(s) or not inked(s):
            continue
        a, b = attach(s.x0, s.y0), attach(s.x1, s.y1)
        if a and b and a is not b and not a.contains(b) and not b.contains(a):
            add(a, b, s, "hough")

    if DEBUG_CONN:
        for a, b, kind, s, src in found.values():
            print("conn", src, (a.x, a.y), (b.x, b.y), [round(v) for v in (s.x0, s.y0, s.x1, s.y1)])
    return [(a, b, kind, s) for a, b, kind, s, _ in found.values()]


def guess_label_edges(root: Node, regions: list[Node], shapes: list[Node], conns: list) -> list:
    """Text sitting in the gap between two boxes, covering most of it, is almost
    always a label on an arrow we couldn't see (hidden under the label). Emit
    it as an edge flagged [guess] instead of leaving it as loose text."""
    leaves = [s for s in shapes if s.kind in ("rect", "ellipse", "diamond")
              and not any(o is not s and s.contains(o, 0) for o in shapes)]
    linked = {frozenset((id(a), id(b))) for a, b, *_ in conns}
    owners = [root] + regions + [s for s in shapes if not any(s is l for l in leaves)]
    out = []
    for owner in owners:
        for w in list(owner.words):
            near = [s for s in leaves if _dist_to_node(w.cx, w.cy, s) < 4 * w.h]
            best = None
            for i, a in enumerate(near):
                for b in near[i + 1:]:
                    if frozenset((id(a), id(b))) in linked:
                        continue
                    dx = (b.x + b.w / 2) - (a.x + a.w / 2)
                    dy = (b.y + b.h / 2) - (a.y + a.h / 2)
                    if not dx and not dy:
                        continue
                    p = _border_point(a, dx, dy)
                    q = _border_point(b, -dx, -dy)
                    seg = Seg(*p, *q)
                    L = seg.length
                    if L < 4 or (q[0] - p[0]) * dx + (q[1] - p[1]) * dy < 0:
                        continue
                    d, t = _point_seg_dist(w.cx, w.cy, seg)
                    if d > 0.6 * w.h or not 0 < t < 1:
                        continue
                    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
                    span = abs(w.w * ux) + abs(w.h * uy)  # word's extent along the gap
                    if span < 0.5 * L or L > 8 * w.h:
                        continue
                    if best is None or d < best[0]:
                        best = (d, a, b, seg)
            if best:
                _, a, b, seg = best
                owner.words.remove(w)
                linked.add(frozenset((id(a), id(b))))
                a.connected = b.connected = True
                out.append((a, b, "--", seg, w.text, {"guess": True}))
    return out


# ---------------------------------------------------------------- tree + text

def build_tree(W: int, H: int, nodes: list[Node], words: list[Word]) -> Node:
    root = Node("rect", 0, 0, W, H, implicit=True)
    placed = [root]
    for n in sorted(nodes, key=lambda n: -n.area):
        parent = min((p for p in placed if p.contains(n)), key=lambda p: p.area)
        parent.kids.append(n)
        placed.append(n)
    for w in words:
        def overlap(p):
            ix = max(0, min(p.x2, w.x2) - max(p.x, w.x))
            iy = max(0, min(p.y2, w.y2) - max(p.y, w.y))
            return ix * iy / max(w.w * w.h, 1)
        inside = [p for p in placed if overlap(p) > 0.8]
        min(inside or [root], key=lambda p: p.area).words.append(w)
    return root


def prune(n: Node) -> tuple[list[Node], list[Word]]:
    """Return what should replace n in its parent: (nodes, loose words)."""
    kids: list[Node] = []
    for k in n.kids:
        nodes, loose = prune(k)
        kids += nodes
        n.words += loose
    n.kids = kids
    items = len(n.words) + len(kids)
    if items == 0 and not n.connected and n.kind not in ("icon", "image"):
        return [], []
    if n.implicit and items <= 1 and not n.connected:
        return kids, n.words  # separates nothing: dissolve
    if not n.words and len(kids) == 1 and not n.connected and (n.implicit or kids[0].area > 0.8 * n.area):
        return kids, []  # same content as its only child
    return [n], []


def merge_lines(words: list[Word]) -> list[Word]:
    """Wrapped lines of one paragraph -> a single multi-line Word."""
    out: list[Word] = []
    for w in sorted(words, key=lambda w: (w.y, w.x)):
        for p in out:
            last_h = p.h / (p.text.count("\n") + 1)
            if (p.bold == w.bold and abs(w.x - p.x) < 0.6 * last_h and 0.75 < w.h / last_h < 1.33
                    and -0.2 * last_h < w.y - p.y2 < 0.35 * last_h):
                x, y = min(p.x, w.x), p.y
                p.text += "\n" + w.text
                p.w, p.h, p.x = max(p.x2, w.x2) - x, w.y2 - y, x
                p.score = min(p.score, w.score)
                break
        else:
            out.append(Word(w.text, w.x, w.y, w.w, w.h, w.score, w.bold))
    return out


def _text(w: Word) -> Text:
    style: dict = {}
    if w.bold:
        style["bold"] = True
    if w.score < 0.85:
        style["?"] = True  # OCR wasn't sure
    return Text(w.text, w.x, w.y, w.w, w.h, style)


def _bbox(items):
    x, y = min(n.x for n in items), min(n.y for n in items)
    return x, y, max(n.x + n.w for n in items) - x, max(n.y + n.h for n in items) - y


def tidy(items: list) -> list:
    """Make a container's children cheaper to read:
    - an icon just left of a label becomes the label's [icon] flag
    - a row of 3+ icons becomes one `icons N` element
    """
    icons = [n for n in items if isinstance(n, Shape) and n.kind == "icon" and not n.children]
    texts = [n for n in items if isinstance(n, Text)]
    used = set()
    for ic in icons:
        for t in texts:
            line_h = t.h / (t.text.count("\n") + 1)
            gap = t.x - (ic.x + ic.w)
            if -0.5 * ic.w <= gap <= max(1.5 * ic.w, 12) and abs((t.y + line_h / 2) - (ic.y + ic.h / 2)) < 0.6 * line_h:
                t.style["icon"] = True
                used.add(id(ic))
                break
    rest = [n for n in items if id(n) not in used]

    # a small number just left of a label is a badge on it (unread counts)
    texts = [n for n in rest if isinstance(n, Text)]
    for b in texts:
        if not (b.text.isdigit() and len(b.text) <= 3):
            continue
        for t in texts:
            if t is b or t.text.isdigit():
                continue
            line_h = t.h / (t.text.count("\n") + 1)
            gap = t.x - (b.x + b.w)
            if b.h <= 0.9 * line_h and -2 <= gap <= 2 * line_h and abs((b.y + b.h / 2) - (t.y + line_h / 2)) < line_h:
                t.style["badge"] = b.text
                used.add(id(b))
                break
    rest = [n for n in rest if id(n) not in used]

    # runs of icons along a row
    icons = sorted((n for n in rest if isinstance(n, Shape) and n.kind == "icon" and not n.children),
                   key=lambda n: (n.y + n.h / 2, n.x))
    others = [n for n in rest if not (isinstance(n, Shape) and n.kind == "icon")]
    rows: list[list] = []
    for ic in icons:
        for row in rows:
            r0 = row[0]
            if abs((ic.y + ic.h / 2) - (r0.y + r0.h / 2)) < 0.5 * max(ic.h, r0.h):
                row.append(ic)
                break
        else:
            rows.append([ic])
    merged = set()
    out_runs = []
    for row in rows:
        row.sort(key=lambda n: n.x)
        run = [row[0]]

        def flush(run):
            if len(run) >= 3:
                x, y, w, h = _bbox(run)
                out_runs.append(Shape("icons", x, y, w, h, style={"_n": len(run)}))
                merged.update(id(n) for n in run)

        for ic in row[1:]:
            prev = run[-1]
            gap = ic.x - (prev.x + prev.w)
            blocked = any(prev.x + prev.w <= o.x + o.w / 2 <= ic.x and
                          abs((o.y + o.h / 2) - (ic.y + ic.h / 2)) < max(ic.h, o.h) for o in others)
            if gap <= 6 * max(ic.w, prev.w) and not blocked:
                run.append(ic)
            else:
                flush(run)
                run = [ic]
        flush(run)
    return [n for n in rest if id(n) not in merged] + out_runs


def tidy_board(board: Board) -> Board:
    """Apply the tidy pass to every container (used for hand-built ground truth)."""
    def rec(items):
        for n in items:
            if isinstance(n, Shape) and n.children:
                n.children = tidy(rec(n.children))
        return items
    board.items = group_columns(tidy(rec(board.items)), min_gap=max(8.0, 0.008 * board.w))
    return board


def group_columns(items: list, min_gap: float, depth: int = 0, ref_h: float | None = None) -> list:
    """Wrap real layout columns (side-by-side stacks of content) in `group`s so
    a reader sees "left nav / feed / sidebar" instead of interleaved rows."""
    if len(items) < 4 or depth > 4:
        return items
    if ref_h is None:  # columns must be tall relative to the whole container, not a band
        ref_h = _bbox(items)[3]
    ycuts = _gaps(items, "y", min_gap)
    if ycuts:  # full-width bands: recurse, don't group
        return [n for band in _split(items, "y", ycuts) for n in group_columns(band, min_gap, depth + 1, ref_h)]
    xcuts = _gaps(items, "x", min_gap)
    if not xcuts:
        return items
    cols = _split(items, "x", xcuts)
    tall = [c for c in cols if len(c) >= 3 and _bbox(c)[3] >= 0.4 * ref_h]
    if len(tall) < 2:
        return items
    out = []
    for c in cols:
        inner = group_columns(c, min_gap, depth + 1, _bbox(c)[3])
        if any(c is t for t in tall):
            out.append(Shape("group", *_bbox(c), children=inner))
        else:
            out.extend(inner)
    return out


def to_board(root: Node, W: int, H: int, connectors) -> Board:
    def texts(words: list[Word]):
        tables, rest = find_tables(words)
        return [_text(w) for w in merge_lines(rest)] + tables

    def build(n: Node) -> Shape:
        tables, rest = find_tables(n.words)
        words = merge_lines(rest)
        style = {"fill": n.fill} if n.fill else {}
        if n.implicit:
            style["stroke"] = "gray"
        label = None
        if n.kind not in ("image", "icon") and not n.implicit and not n.kids and not tables and len(words) == 1:
            w = words[0]
            if abs(w.cx - (n.x + n.w / 2)) < 0.15 * n.w and abs(w.cy - (n.y + n.h / 2)) < 0.2 * n.h:
                label, words = w.text, []
        s = Shape(n.kind, n.x, n.y, n.w, n.h, label=label, style=style)
        s.children = tidy([_text(w) for w in words] + tables + [build(k) for k in n.kids])
        n.shape = s
        return s

    items = texts(root.words)
    items += [build(k) for k in root.kids]
    items = group_columns(tidy(items), min_gap=max(8.0, 0.008 * W))
    edges = [Edge(a.shape, b.shape, c[4], c[2], dict(c[5]) if len(c) > 5 else {})
             for c in connectors for a, b in [(c[0], c[1])]
             if a.shape is not None and b.shape is not None]
    return Board(W, H, items, edges)


# ---------------------------------------------------------------- main

@dataclass
class Result:
    timings: dict
    board: Board
    words: list[Word]
    shapes: list[Node]
    regions: list[Node]
    connectors: list


def convert(img: np.ndarray, raw_words: list[Word] | None = None) -> Result:
    """Screenshot -> Board. Pass raw_words to supply OCR yourself (e.g. the
    video runner's incremental OCR); otherwise the whole image is OCR'd."""
    import time
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name):
        nonlocal t0
        t = time.perf_counter()
        timings[name] = round((t - t0) * 1000)
        t0 = t

    H, W = img.shape[:2]
    raw = [Word(**w.__dict__) for w in raw_words] if raw_words is not None else ocr(img)
    lap("ocr")
    glyphs = all_words(raw)
    # icons misread as a glyph or two, whatever OCR's confidence
    labels_ok = [w for w in glyphs if w.score >= 0.6]
    _, guessed = misread_icons(glyphs, [])
    guessed_ids = {(g.x, g.y) for g in guessed}
    words = clean_words([Word(**w.__dict__) for w in raw if (w.x, w.y) not in guessed_ids])
    mark_bold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), words)
    clean = bridge_lines(erase_text(img, glyphs), glyphs)
    lap("erase_text")
    edges = edge_map(clean)
    lap("edges")
    shapes = find_shapes(edges)
    lap("shapes")
    regions = find_regions(edges, shapes)
    lap("regions")

    bg_bgr = np.median(clean.reshape(-1, 3), axis=0)
    bg = bg_bgr[::-1].astype(float)
    for s in shapes:
        if is_photo(img, s):
            s.kind = "image"
        else:
            s.fill = shape_fill(clean, s, bg)
    # contours inside a word's box are pieces of its letters ("0", "6", "8" holes)
    shapes = [s for s in shapes if not any(
        w.x - 4 <= s.x and w.y - 4 <= s.y and s.x2 <= w.x2 + 4 and s.y2 <= w.y2 + 4 for w in words)]
    # contours found inside a photo are part of the picture
    images = [s for s in shapes if s.kind == "image"]
    shapes = [s for s in shapes if not any(i is not s and i.contains(s, 0) for i in images)]

    # a glyph that is the only content of a small shape is its label, not an icon
    guessed = [g for g in guessed if not any(sh.contains(g, 2) and sh.w < 4 * g.w + 40 for sh in shapes)]
    words += [Word(w.text, w.x, w.y, w.w, w.h, w.score) for w in labels_ok
              if (w.x, w.y) in guessed_ids and not any(g.x == w.x and g.y == w.y for g in guessed)]
    lap("classify")
    conns = find_connectors(img, clean, shapes, words)
    lap("connectors")
    for a, b, *_ in conns:
        a.connected = b.connected = True
    med_h = float(np.median([w.h for w in words])) if words else 20.0

    # Diagram nodes with no text are usually a lone glyph OCR detection missed
    # (a "1" in a circle): recognise just those interiors.
    inside, filled = ocr_inside(img, [s for s in shapes if s.connected], words, med_h)
    words += inside
    # contours nested in a shape whose text we just read are pieces of that glyph
    glyph_bits = {id(s) for s in shapes if any(f is not s and f.contains(s, 0) for f in filled)}
    shapes = [s for s in shapes if id(s) not in glyph_bits]
    conns = [c for c in conns if id(c[0]) not in glyph_bits and id(c[1]) not in glyph_bits]
    lap("glyphs")
    # small textless circles/squares (avatars, emoji, buttons) are icons themselves
    for sh in shapes:
        small = max(sh.w, sh.h) <= 3 * med_h
        if small and not sh.connected and sh.kind in ("ellipse", "rect") and not any(
                sh.contains(w, 0) for w in words):
            sh.kind, sh.fill = "icon", None
    icons = guessed + find_icons(clean, glyphs + inside, shapes, conns, med_h)
    icons = [i for i in icons if not any(sh.kind == "icon" and sh.contains(i, 2) for sh in shapes)]

    lap("icons")
    root = build_tree(W, H, shapes + regions + icons, words)

    # Free text sitting on a connector becomes its label.
    labelled = []
    loose = [(w, owner) for owner in [root] + regions for w in owner.words]
    for a, b, kind, s in conns:
        best = None
        for w, owner in loose:
            d, t = _point_seg_dist(w.cx, w.cy, s)
            if d < max(20, w.h) and 0.1 < t < 0.9 and (best is None or d < best[0]):
                best = (d, w, owner)
        label = None
        if best:
            _, w, owner = best
            owner.words.remove(w)
            loose.remove((w, owner))
            label = w.text
        labelled.append((a, b, kind, s, label))

    labelled += guess_label_edges(root, regions, shapes, labelled)
    prune(root)
    board = to_board(root, W, H, labelled)
    lap("tree+board")
    return Result(timings, board, words, shapes + icons, regions, labelled)


def debug_overlay(img: np.ndarray, res: Result) -> np.ndarray:
    out = (img * 0.45 + 255 * 0.55).astype(np.uint8)
    for r in res.regions:
        cv2.rectangle(out, (r.x, r.y), (r.x2, r.y2), (200, 120, 0), 2)
    for s in res.shapes:
        color = {"rect": (0, 140, 0), "ellipse": (0, 0, 220), "diamond": (180, 0, 180),
                 "image": (0, 200, 255), "icon": (255, 0, 0)}[s.kind]
        cv2.rectangle(out, (s.x, s.y), (s.x2, s.y2), color, 2)
    for w in res.words:
        cv2.rectangle(out, (w.x, w.y), (w.x2, w.y2), (0, 165, 255), 1)
    for a, b, kind, s, label, *_ in res.connectors:
        cv2.line(out, (int(s.x0), int(s.y0)), (int(s.x1), int(s.y1)), (255, 0, 255), 3)
        if kind != "--":
            cv2.circle(out, (int(s.x1), int(s.y1)), 7, (255, 0, 255), -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--debug", action="store_true", help="also write a debug overlay to pipeline/out/")
    ap.add_argument("--width", type=int, default=1000, help="normalise board width (0 = keep pixels)")
    ap.add_argument("--geometry", choices=["full", "boxes", "none"], default="full",
                    help="how much geometry to write (full is needed for faithful rendering)")
    args = ap.parse_args()
    for path in args.images:
        img = cv2.imread(str(path))
        res = convert(img)
        out = path.with_suffix(".pred.board")
        board = scale_board(res.board, args.width) if args.width else res.board
        out.write_text(serialize(board, args.geometry))
        print(f"{path} -> {out.name}: {len(res.words)} text, {len(res.shapes)} shapes, "
              f"{len(res.regions)} regions, {len(res.connectors)} connectors | "
              f"{sum(res.timings.values())}ms {res.timings}")
        if args.debug:
            dbg = ROOT / "out" / f"{path.parent.name}-{path.stem}.debug.png"
            dbg.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(dbg), debug_overlay(img, res))


if __name__ == "__main__":
    main()

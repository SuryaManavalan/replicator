"""Structured blocks: tables and code.

These are data, not decoration, so they get dedicated detectors that run
before generic shape detection:

- Ruled tables: the grid is rebuilt from the actual horizontal/vertical lines;
  every word lands in exactly one cell. Tables with row rules only take their
  columns from whitespace gutters.
- Unruled tables: runs of evenly spaced rows whose columns are separated by
  whitespace gutters (works for left/right/centre aligned columns).
- Code: a column of consecutive line numbers marks a code block; the text on
  those rows is the code, with indentation and spacing rebuilt from the
  character width.

All sizes are relative to the median text height (`med_h`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .board import Code, Table


@dataclass
class Found:
    block: Table | Code
    words: list  # the words it consumed
    x: int
    y: int
    w: int
    h: int
    line_boxes: list | None = None  # code: per line (x0, y0, x1, y1) of its text, or None if blank
    indent_cw: float = 0.0  # code: character width
    indent_base: float = 0.0  # code: x of column 0
    bounds: tuple | None = None  # table: (column xs, row ys) cell boundaries


def _cy(w):
    return w.y + w.h / 2


def _cx(w):
    return w.x + w.w / 2


# ---------------------------------------------------------------- code

NUM = re.compile(r"\d{1,5}")


def find_code_blocks(words: list, med_h: float) -> list[Found]:
    """Line-number gutters: >= 3 consecutive integers, stacked with a steady
    pitch and a shared right edge. The words to their right on the same rows
    are the code."""
    nums = [w for w in words if NUM.fullmatch(w.text)]
    tol = 0.8 * med_h
    cols: list[list] = []
    for w in sorted(nums, key=lambda w: w.x + w.w):
        for c in cols:
            if abs((c[0].x + c[0].w) - (w.x + w.w)) < tol:
                c.append(w)
                break
        else:
            cols.append([w])

    found: list[Found] = []
    used: set[int] = set()
    for col in cols:
        col.sort(key=_cy)
        runs, run = [], [col[0]]
        for w in col[1:]:
            prev = run[-1]
            k = int(w.text) - int(prev.text)
            dy = _cy(w) - _cy(prev)
            pitch = (_cy(run[-1]) - _cy(run[0])) / (int(run[-1].text) - int(run[0].text)) if len(run) > 1 else None
            ok = k >= 1 and k <= 3 and dy > 0 and (pitch is None and dy < 3 * med_h * k
                                                  or pitch is not None and abs(dy - k * pitch) < 0.35 * pitch)
            if ok:
                run.append(w)
            else:
                runs.append(run)
                run = [w]
        runs.append(run)
        for run in runs:
            if len(run) < 3:
                continue
            f = _code_from_gutter(run, [w for w in words if id(w) not in used and w not in run], med_h)
            if f:
                found.append(f)
                used.update(id(w) for w in f.words)
    return found


def _code_from_gutter(run: list, words: list, med_h: float) -> Found | None:
    first, last = int(run[0].text), int(run[-1].text)
    pitch = (_cy(run[-1]) - _cy(run[0])) / max(last - first, 1)
    gutter_x2 = max(w.x + w.w for w in run)
    y0_run = _cy(run[0]) - (int(run[0].text) - first) * pitch

    def has_code_at(y):
        return any(w.x >= gutter_x2 and abs(_cy(w) - y) < 0.45 * pitch and w.x - gutter_x2 < 60 * med_h for w in words)

    # OCR often misses a lone digit ("1"): extend over rows that hold code
    while first > 1 and has_code_at(y0_run - pitch):
        first -= 1
        y0_run -= pitch
    while has_code_at(y0_run + (last - first + 1) * pitch):
        last += 1
    y_of = {first + i: y0_run + i * pitch for i in range(last - first + 1)}
    # code words: right of the gutter, on one of the numbered rows
    rows: dict[int, list] = {n: [] for n in y_of}
    for w in words:
        if w.x < gutter_x2 or w.x - gutter_x2 > 60 * med_h:
            continue
        n = min(y_of, key=lambda n: abs(y_of[n] - _cy(w)))
        if abs(y_of[n] - _cy(w)) < 0.45 * pitch:
            rows[n].append(w)
    code_words = [w for ws in rows.values() for w in ws]
    if not code_words:
        return None
    # character width (code is monospaced) and where column 0 is
    cw = float(np.median([w.w / max(len(w.text), 1) for w in code_words if len(w.text) >= 4] or
                         [w.w / max(len(w.text), 1) for w in code_words]))
    base = min(sorted(ws, key=lambda w: w.x)[0].x for ws in rows.values() if ws)
    lines, boxes = [], []
    for n in sorted(rows):
        ws = sorted(rows[n], key=lambda w: w.x)
        if not ws:
            lines.append("")
            boxes.append(None)
            continue
        boxes.append((ws[0].x, min(w.y for w in ws), max(w.x + w.w for w in ws), max(w.y + w.h for w in ws)))
        s = " " * max(0, round((ws[0].x - base) / cw)) + ws[0].text
        for a, b in zip(ws, ws[1:]):
            s += " " * max(1, round((b.x - (a.x + a.w)) / cw)) + b.text
        lines.append(s.rstrip())
    # numbered rows whose text is split by several wide gaps are a table with a
    # rank/row-number column, not code
    wide = sum(1 for l in lines if len(re.findall(r"\S {6,}\S", l)) >= 2)
    if wide > 0.4 * max(1, sum(1 for l in lines if l)):
        return None
    xs = [w.x for w in run] + [w.x for w in code_words]
    x2 = max(w.x + w.w for w in run + code_words)
    y0 = min(w.y for w in run + code_words)
    y2 = max(w.y + w.h for w in run + code_words)
    code = Code(min(xs), y0, x2 - min(xs), y2 - y0, lines, start=first)
    return Found(code, run + code_words, min(xs), y0, x2 - min(xs), y2 - y0, boxes, cw, base)


def reread_empty_cells(img: np.ndarray, found: list[Found], rec, med_h: float, limit: int = 300) -> None:
    """OCR detection misses lone glyphs (a "1" in a row-number cell). Re-read
    the empty cells of ruled tables directly; empty cells read as nothing."""
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    H, W = img.shape[:2]
    jobs = []
    for f in found:
        if not isinstance(f.block, Table) or not f.bounds:
            continue
        xs, ys = f.bounds
        if len(xs) != len(f.block.rows[0]) + 1 or len(ys) != len(f.block.rows) + 1:
            continue
        for i, row in enumerate(f.block.rows):
            for j, v in enumerate(row):
                if v:
                    continue
                m = max(2, int(0.12 * med_h))
                crop = img[max(int(ys[i]) + m, 0):min(int(ys[i + 1]) - m, H), max(int(xs[j]) + m, 0):min(int(xs[j + 1]) - m, W)]
                if crop.size and crop.shape[0] >= 6 and crop.shape[1] >= 6 and crop.std() > 8:  # skip blank cells
                    jobs.append((f, i, j, crop))
    jobs = jobs[:limit]
    if not jobs:
        return
    out = rec(TextRecInput(img=[j[3] for j in jobs]))
    for (f, i, j, _), txt, score in zip(jobs, out.txts, out.scores):
        txt = txt.strip()
        if txt and score >= 0.8 and any(ch.isalnum() for ch in txt):
            f.block.rows[i][j] = txt


def reread_code_lines(img: np.ndarray, found: list[Found], rec) -> None:
    """Syntax colouring splits a code line into many small OCR boxes and the
    punctuation at their edges gets lost. Re-recognise each whole line as one
    tight crop (batched); indentation still comes from the line's position."""
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    jobs = []
    H, W = img.shape[:2]
    for f in found:
        if not isinstance(f.block, Code) or not f.line_boxes:
            continue
        for i, b in enumerate(f.line_boxes):
            if b is None:
                continue
            x0, y0, x1, y1 = b
            cw = f.indent_cw or 0.5 * (y1 - y0)
            vy = max(2, int(0.12 * (y1 - y0)))
            crop = img[max(y0 - vy, 0):min(y1 + vy, H),
                       max(int(x0 - 0.3 * cw - 1), 0):min(int(x1 + 0.6 * cw + 3), W)]
            if crop.size:
                jobs.append((f, i, x0, crop))
    if not jobs:
        return
    out = rec(TextRecInput(img=[j[3] for j in jobs]))
    for (f, i, x0, _), txt, score in zip(jobs, out.txts, out.scores):
        txt = txt.strip()
        if not txt or score < 0.5:
            continue
        indent = " " * max(0, round((x0 - f.indent_base) / f.indent_cw)) if f.indent_cw else ""
        old = f.block.lines[i].strip()
        if len(old) > len(txt) and txt.replace(" ", "") in old.replace(" ", ""):
            continue  # the re-read lost characters (very short lines): keep the first reading
        f.block.lines[i] = indent + txt


# ---------------------------------------------------------------- tables

def _gutters(rows: list[list], x0: int, x1: int, min_gap: float) -> list[tuple[int, int]]:
    """Column intervals separated by vertical whitespace shared by (almost)
    every row."""
    span = x1 - x0
    if span <= 0:
        return []
    cover = np.zeros(span, np.int32)
    for row in rows:
        mark = np.zeros(span, bool)
        for w in row:
            a, b = max(w.x - x0, 0), min(w.x + w.w - x0, span)
            mark[a:b] = True
        cover += mark
    busy = cover > max(1, 0.15 * len(rows))  # a single spanning row doesn't close a gutter
    cols, start, gap = [], None, 0
    for i, b in enumerate(busy):
        if b:
            if start is None:
                start = i
            elif gap >= min_gap:
                cols.append((x0 + start, x0 + i - gap))
                start = i
            gap = 0
        elif start is not None:
            gap += 1
    if start is not None:
        cols.append((x0 + start, x0 + span - gap))
    return cols


def _cells(rows: list[list], bounds: list[float]) -> list[list[str]]:
    """Words -> cell strings, by which column band holds each word's centre.
    A word spanning several bands (OCR ran two cells together) is split by
    character count at spaces."""
    out = []
    for row in rows:
        cells = [[] for _ in range(len(bounds) - 1)]
        for w in sorted(row, key=lambda w: (round(_cy(w) / max(w.h, 1)), w.x)):
            spanned = [i for i in range(len(bounds) - 1)
                       if min(w.x + w.w, bounds[i + 1]) - max(w.x, bounds[i]) > 0.25 * (bounds[i + 1] - bounds[i])]
            if len(spanned) > 1 and " " in w.text:
                parts = _split_by_bands(w, bounds, spanned)
                for i, t in zip(spanned, parts):
                    if t:
                        cells[i].append(t)
                continue
            c = max(0, min(len(cells) - 1, int(np.searchsorted(bounds, _cx(w))) - 1))
            cells[c].append(w.text)
        out.append([" ".join(c).strip() for c in cells])
    return out


def _split_by_bands(w, bounds, spanned) -> list[str]:
    tokens = w.text.split(" ")
    per = w.w / max(len(w.text), 1)
    parts = [[] for _ in spanned]
    pos = w.x
    for t in tokens:
        mid = pos + per * len(t) / 2
        k = max(0, min(len(spanned) - 1, int(np.searchsorted([bounds[i + 1] for i in spanned], mid))))
        parts[k].append(t)
        pos += per * (len(t) + 1)
    return [" ".join(p) for p in parts]


def _group_rows(words: list, med_h: float) -> list[list]:
    rows: list[list] = []
    for w in sorted(words, key=_cy):
        if rows and abs(_cy(rows[-1][0]) - _cy(w)) < 0.45 * min(rows[-1][0].h, w.h, med_h * 1.2):
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


def _line_segments(mask: np.ndarray, horizontal: bool, max_thick: int):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for x, y, w, h, _a in stats[1:]:
        if horizontal and h <= max_thick:
            segs.append((x, x + w, y + h / 2))
        elif not horizontal and w <= max_thick:
            segs.append((y, y + h, x + w / 2))
    return segs


def _covered_lines(segs: list, extent: float, pos_tol: float = 2.5) -> list[float]:
    """Positions where line segments together cover at least half the extent
    (a rule may be broken into several pieces)."""
    groups: list[list] = []
    for a0, a1, p in sorted(segs, key=lambda t: t[2]):
        if groups and abs(groups[-1][0][2] - p) <= pos_tol:
            groups[-1].append((a0, a1, p))
        else:
            groups.append([(a0, a1, p)])
    out = []
    for g in groups:
        covered = sum(a1 - a0 for a0, a1, _ in g)
        if covered >= 0.5 * extent:
            out.append(sum(p for *_, p in g) / len(g))
    return _merge_positions(out, 4)


def _split_tall(ys: list[float], words: list, med_h: float, cols: list[float] | None = None) -> list[float]:
    """A band holding k text rows that is ~k typical rows tall lost its
    dividing rules (faint against a fill, broken by compression): split it.
    Only if every text row in it is a *full* row: a header whose cells wrap
    onto a second line has a sparse second line, and stays one row."""
    if len(ys) < 3:
        return ys
    h = float(np.median(np.diff(ys)))

    def col_of(w):
        return int(np.searchsorted(cols, _cx(w))) if cols else 0

    out = [ys[0]]
    for a, b in zip(ys, ys[1:]):
        k = round((b - a) / h)
        rows = _group_rows([w for w in words if a <= _cy(w) < b], med_h)
        if k >= 2 and len(rows) >= 2 and abs((b - a) - k * h) < 0.25 * h:
            occupied = {col_of(w) for r in rows for w in r}
            full = all(len({col_of(w) for w in r}) >= 0.6 * len(occupied) for r in rows)
            if full or not cols:
                out += [a + i * (b - a) / k for i in range(1, k)]
        out.append(b)
    return out


def _split_wide(xs: list[float], ys: list[float], words: list, med_h: float) -> list[float]:
    """A column whose rule was lost (cursor over it, compression): if its text
    is split by a clear gutter in most rows, split the column there."""
    out = [xs[0]]
    for a, b in zip(xs, xs[1:]):
        band = [w for w in words if a <= _cx(w) < b]
        rows = [[w for w in band if ys[i] <= _cy(w) < ys[i + 1]] for i in range(len(ys) - 1)]
        rows = [r for r in rows if r]
        cols = _gutters(rows, int(a), int(b), max(8, 1.2 * med_h)) if len(rows) >= 3 else []
        if len(cols) >= 2:
            both = sum(1 for r in rows if {min(range(len(cols)), key=lambda k: abs((cols[k][0] + cols[k][1]) / 2 - _cx(w)))
                                           for w in r} == set(range(len(cols))))
            if both >= 0.5 * len(rows):
                out += [(c[1] + d[0]) / 2 for c, d in zip(cols, cols[1:])]
        out.append(b)
    return out


def _outer_bands(bounds: list[float], words: list, med_h: float, along_y=False,
                 cols: list[float] | None = None) -> list[float]:
    """Add one band beyond the outermost rules when text sits there in line
    with the table (e.g. a row-number column with no rule on its left). An
    extra row must fit the columns: a caption running across them is not a row."""
    if len(bounds) < 2:
        return bounds

    def fits(ws):
        if not along_y or not cols:
            return True
        crossing = sum(1 for w in ws if any(w.x + 0.15 * w.w < c < w.x + 0.85 * w.w for c in cols[1:-1]))
        return crossing <= 0.2 * len(ws)
    pos = (lambda w: _cy(w)) if along_y else (lambda w: _cx(w))
    lo = (lambda w: w.y) if along_y else (lambda w: w.x)
    hi = (lambda w: w.y + w.h) if along_y else (lambda w: w.x + w.w)
    size = float(np.median(np.diff(bounds)))
    before = [w for w in words if bounds[0] - 1.5 * size < pos(w) < bounds[0]]
    after = [w for w in words if bounds[-1] < pos(w) < bounds[-1] + 1.5 * size]
    out = list(bounds)
    if len(before) >= 2 and fits(before):
        out = [min(lo(w) for w in before) - 0.2 * med_h] + out
    if len(after) >= 2 and fits(after):
        out = out + [max(hi(w) for w in after) + 0.2 * med_h]
    return out


def _join_collinear(segs: list, pos_tol: float, max_gap: float) -> list:
    """Segments (start, end, pos) on the same line with small gaps -> one."""
    out = []
    for a0, a1, p in sorted(segs, key=lambda t: (round(t[2] / pos_tol), t[0])):
        if out and abs(out[-1][2] - p) <= pos_tol and a0 - out[-1][1] <= max_gap:
            b0, b1, q = out[-1]
            out[-1] = (b0, max(b1, a1), (q + p) / 2)
        else:
            out.append((a0, a1, p))
    return out


def _merge_positions(vals: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def find_grid_tables(clean_gray: np.ndarray, words: list, med_h: float) -> list[Found]:
    """Tables drawn with rules. Lines are found in the text-erased image, so
    glyph strokes can't pose as rules."""
    H, W = clean_gray.shape
    bg = cv2.medianBlur(clean_gray, 31)
    # faint rules (dark-on-dark separators) are only a few levels off the
    # background (7 levels); noise can't pass the long straight-run filter below
    ink = (cv2.absdiff(clean_gray, bg) > 7).astype(np.uint8)
    lh = max(40, int(4 * med_h))
    lv = max(20, int(1.4 * med_h))
    hmask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, lh), np.uint8))
    vmask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((lv, 1), np.uint8))
    thick = max(3, int(0.25 * med_h))
    # compression and nearby text break rules into pieces: join collinear ones
    hsegs = _join_collinear(_line_segments(hmask, True, thick), 2.5, 2 * med_h)
    vsegs = _join_collinear(_line_segments(vmask, False, thick), 2.5, 2 * med_h)
    union = cv2.dilate(hmask | vmask, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    found: list[Found] = []
    taken: set[int] = set()
    for gx, gy, gw, gh, _a in stats[1:]:
        if gw < 4 * med_h or gh < 2 * med_h:
            continue
        gx2, gy2 = gx + gw, gy + gh
        hl = _covered_lines([(x0, x1, y) for x0, x1, y in hsegs if gx - 3 <= x0 and x1 <= gx2 + 3 and gy - 3 <= y <= gy2 + 3], gw)
        vl = _covered_lines([(y0, y1, x) for y0, y1, x in vsegs if gy - 3 <= y0 and y1 <= gy2 + 3 and gx - 3 <= x <= gx2 + 3], gh)
        inside = [w for w in words if id(w) not in taken and gx - 2 <= _cx(w) <= gx2 + 2 and gy - 2 <= _cy(w) <= gy2 + 2]
        if len(hl) >= 2 and len(vl) >= 2:
            xs = _split_wide(_outer_bands(vl, inside, med_h), hl, inside, med_h)
            ys = _split_tall(hl, inside, med_h, cols=xs)
            ys = _outer_bands(ys, inside, med_h, along_y=True, cols=xs)
            rows_words = [[w for w in inside if ys[i] <= _cy(w) < ys[i + 1]] for i in range(len(ys) - 1)]
            grid = _cells(rows_words, xs)
            style = {"grid": True}
        elif len(hl) >= 3:
            # row rules only: rows from the rules, columns from gutters
            ys = hl
            rows_words = [[w for w in inside if ys[i] <= _cy(w) < ys[i + 1]] for i in range(len(ys) - 1)]
            cols = _gutters([r for r in rows_words if r], gx, gx2, max(8, 0.8 * med_h))
            if len(cols) < 2:
                continue
            xs = [cols[0][0]] + [(a[1] + b[0]) / 2 for a, b in zip(cols, cols[1:])] + [cols[-1][1]]
            grid = _cells(rows_words, xs)
            style = {}
        else:
            continue
        # trim empty rows/columns at the edges (sheet chrome, unused grid)
        while grid and not any(grid[-1]):
            grid.pop(); rows_words.pop(); ys = ys[:-1]
        while grid and not any(grid[0]):
            grid.pop(0); rows_words.pop(0); ys = ys[1:]
        if not grid:
            continue
        ncol = len(grid[0])
        while ncol > 1 and not any(r[ncol - 1] for r in grid):
            grid = [r[:-1] for r in grid]; ncol -= 1; xs = xs[:-1]
        while ncol > 1 and not any(r[0] for r in grid):
            grid = [r[1:] for r in grid]; ncol -= 1; xs = xs[1:]
        filled = sum(1 for r in grid for c in r if c)
        if len(grid) < 2 or ncol < 2 or filled < 4:
            continue
        heights = np.diff(ys)
        if np.median(heights) > 3.5 * med_h or (heights > 5 * med_h).mean() > 0.3:
            continue  # tall cells: page layout boxes (cards, panels), not a table
        used = [w for r in rows_words for w in r]
        x0, y0 = int(xs[0]), int(ys[0])
        tb = Table(x0, y0, int(xs[-1] - xs[0]) if len(xs) > 1 else gw, int(ys[-1] - ys[0]) if len(ys) > 1 else gh,
                   [float(x) for x in xs[:-1]], grid, style)
        found.append(Found(tb, used, int(tb.x), int(tb.y), int(tb.w), int(tb.h), bounds=(list(xs), list(ys))))
        taken.update(id(w) for w in used)

    # Row rules only (no vertical lines to connect them): stacks of parallel
    # horizontal rules sharing their extent, with steady spacing.
    free = [sg for sg in hsegs if not any(f.x - 4 <= sg[0] and sg[1] <= f.x + f.w + 4 and f.y - 4 <= sg[2] <= f.y + f.h + 4
                                          for f in found)]
    stacks: list[list] = []
    for sg in sorted(free, key=lambda t: t[2]):
        for st in stacks:
            a = st[-1]
            if abs(a[0] - sg[0]) <= 6 and abs(a[1] - sg[1]) <= 6 and sg[2] - a[2] <= 4.5 * med_h:
                st.append(sg)
                break
        else:
            stacks.append([sg])
    for st in stacks:
        if len(st) < 3:
            continue
        x0, x1 = min(t[0] for t in st), max(t[1] for t in st)
        ys = _merge_positions([t[2] for t in st], 4)
        gaps = np.diff(ys)
        pitch = float(np.median(gaps))
        if pitch > 3.5 * med_h:
            continue
        pool = [w for w in words if id(w) not in taken and x0 - 2 <= _cx(w) <= x1 + 2]
        # a header row above the first rule / a last row below the last rule
        # (only if it looks like a table row: at least two separate words)
        if sum(1 for w in pool if ys[0] - pitch <= _cy(w) < ys[0]) >= 2:
            ys = [ys[0] - pitch] + ys
        if sum(1 for w in pool if ys[-1] < _cy(w) <= ys[-1] + pitch) >= 2:
            ys = ys + [ys[-1] + pitch]
        rows_words = [[w for w in pool if ys[i] <= _cy(w) < ys[i + 1]] for i in range(len(ys) - 1)]
        rows_words = [r for r in rows_words if r]
        if len(rows_words) < 2:
            continue
        cols = _gutters(rows_words, x0, x1, max(8, 0.8 * med_h))
        if len(cols) < 2:
            continue
        xs = [cols[0][0]] + [(a[1] + b[0]) / 2 for a, b in zip(cols, cols[1:])] + [cols[-1][1]]
        grid = _cells(rows_words, xs)
        if sum(1 for r in grid if sum(1 for c in r if c) >= 2) < 2:
            continue
        used = [w for r in rows_words for w in r]
        ty0 = int(min(w.y for w in used))
        tb = Table(int(x0), ty0, int(x1 - x0), int(max(w.y + w.h for w in used) - ty0),
                   [float(x) for x in xs[:-1]], grid, {})
        found.append(Found(tb, used, int(tb.x), int(tb.y), int(tb.w), int(tb.h)))
        taken.update(id(w) for w in used)
    return found


def find_aligned_tables(words: list, med_h: float) -> tuple[list[Table], list]:
    """Unruled tables: >= 3 evenly spaced rows whose words fall into >= 2
    columns separated by whitespace gutters."""
    if len(words) < 6:
        return [], words
    rows = _group_rows(words, med_h)
    tables, used = [], set()
    i = 0
    while i < len(rows):
        if len(rows[i]) < 2:
            i += 1
            continue
        block = [rows[i]]
        pitch = None
        for r in rows[i + 1:]:
            dy = _cy(r[0]) - _cy(block[-1][0])
            if pitch is None:
                if dy > 3.5 * med_h:
                    break
                pitch = dy
            elif not 0.7 * pitch <= dy <= 1.35 * pitch:
                break
            block.append(r)
        best = None
        for end in range(len(block), 2, -1):  # longest prefix that is a table
            b = block[:end]
            x0 = min(w.x for r in b for w in r)
            x1 = max(w.x + w.w for r in b for w in r)
            cols = _gutters(b, x0, x1, max(8, 0.9 * med_h))
            if len(cols) < 2:
                continue
            bounds = [cols[0][0]] + [(a[1] + c[0]) / 2 for a, c in zip(cols, cols[1:])] + [cols[-1][1]]
            grid = _cells(b, bounds)
            multi = sum(1 for r in grid if sum(1 for c in r if c) >= 2)
            if multi >= 0.6 * len(grid) and multi >= 3:
                best = (b, bounds, grid)
                break
        if best:
            b, bounds, grid = best
            ws = [w for r in b for w in r]
            x, y = min(w.x for w in ws), min(w.y for w in ws)
            size = round(float(np.median([w.h for w in ws])) / 1.25)
            tables.append(Table(x, y, max(w.x + w.w for w in ws) - x, max(w.y + w.h for w in ws) - y,
                                [float(c) for c in bounds[:-1]], grid, {"size": size}))
            used.update(id(w) for w in ws)
            i += len(b)
        else:
            i += 1
    return tables, [w for w in words if id(w) not in used]

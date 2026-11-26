// Board AST -> Excalidraw element skeletons (fed to convertToExcalidrawElements).

import type { Board, BoardNode, EdgeNode, Geom, ShapeNode, Style } from "./parse";

export const LINE_HEIGHT = 1.25;
const LABEL_SIZE = 20;
const PAD = 20;
const GAP = 16;
const TITLE_H = 44;

export const STROKE: Record<string, string> = {
  black: "#1e1e1e", gray: "#868e96", red: "#e03131", pink: "#c2255c", violet: "#6741d9",
  blue: "#1971c2", cyan: "#0c8599", teal: "#099268", green: "#2f9e44", yellow: "#f08c00",
  orange: "#e8590c",
};
export const FILL: Record<string, string> = {
  black: "#343a40", gray: "#e9ecef", red: "#ffc9c9", pink: "#fcc2d7", violet: "#d0bfff",
  blue: "#a5d8ff", cyan: "#99e9f2", teal: "#96f2d7", green: "#b2f2bb", yellow: "#ffec99",
  orange: "#ffd8a8",
};

export interface RenderOptions {
  sketchy: boolean; // hand-drawn look vs clean
}

type Box = Required<Geom>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Skeleton = any;

const color = (table: Record<string, string>, v: string | true | undefined) =>
  typeof v === "string" ? (v.startsWith("#") ? v : table[v]) : undefined;

function measureText(text: string, fontSize: number) {
  const lines = text.split("\n");
  return {
    w: Math.max(...lines.map((l) => l.length)) * fontSize * 0.55,
    h: lines.length * fontSize * LINE_HEIGHT,
  };
}

// ---------- auto layout for nodes without geometry ----------

function autoSize(n: BoardNode): { w: number; h: number } {
  if (n.type === "text") {
    if (n.geom?.w && n.geom?.h) return { w: n.geom.w, h: n.geom.h };
    return measureText(n.text, LABEL_SIZE);
  }
  if (n.type === "table") {
    if (n.geom?.w && n.geom?.h) return { w: n.geom.w, h: n.geom.h };
    const size = Number(n.style.size) || 16;
    const widths = n.rows[0]?.map((_, c) => Math.max(...n.rows.map((r) => measureText(r[c] ?? "", size).w))) ?? [];
    return { w: widths.reduce((a, b) => a + b + 24, 0), h: n.rows.length * size * 2 };
  }
  const kids = n.children.map(autoSize);
  if (n.geom?.w && n.geom?.h) return { w: n.geom.w, h: n.geom.h };
  if (kids.length) {
    return {
      w: Math.max(160, ...kids.map((k) => k.w + 2 * PAD)),
      h: (n.label ? TITLE_H : PAD) + kids.reduce((s, k) => s + k.h + GAP, 0) - GAP + PAD,
    };
  }
  const label = n.label ? measureText(n.label, LABEL_SIZE) : { w: 0, h: 0 };
  return { w: Math.max(160, label.w + 40), h: Math.max(80, label.h + 30) };
}

function place(nodes: BoardNode[], originX: number, originY: number, flow: "row" | "column", out: Map<BoardNode, Box>) {
  let cx = originX;
  let cy = originY;
  for (const n of nodes) {
    const size = autoSize(n);
    let box: Box;
    if (n.geom) {
      box = { x: n.geom.x, y: n.geom.y, w: size.w, h: size.h };
    } else {
      box = { x: cx, y: cy, ...size };
      if (flow === "row") cx += size.w + 60;
      else cy += size.h + GAP;
    }
    out.set(n, box);
    if (n.type === "shape" && n.children.length) {
      place(n.children, box.x + PAD, box.y + (n.label ? TITLE_H : PAD), "column", out);
    }
  }
}

function layout(board: Board): Map<BoardNode, Box> {
  const out = new Map<BoardNode, Box>();
  // unpositioned top-level nodes go in a row below positioned ones
  const positioned = board.items.filter((n) => n.geom);
  place(positioned, 0, 0, "row", out);
  let bottom = 40;
  for (const b of out.values()) bottom = Math.max(bottom, b.y + b.h + 60);
  place(board.items.filter((n) => !n.geom), 40, bottom, "row", out);
  return out;
}

// ---------- edges ----------

function borderPoint(kind: ShapeNode["kind"], b: Box, dx: number, dy: number) {
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2, hw = b.w / 2, hh = b.h / 2;
  let t: number;
  if (kind === "ellipse") t = 1 / Math.hypot(dx / hw, dy / hh);
  else if (kind === "diamond") t = 1 / (Math.abs(dx) / hw + Math.abs(dy) / hh);
  else t = Math.min(dx ? hw / Math.abs(dx) : Infinity, dy ? hh / Math.abs(dy) : Infinity);
  return [cx + dx * t, cy + dy * t];
}

// ---------- conversion ----------

export interface ConvertResult {
  skeletons: Skeleton[];
  warnings: { line: number; message: string }[];
}

export function toSkeletons(board: Board, opts: RenderOptions): ConvertResult {
  const boxes = layout(board);
  const skeletons: Skeleton[] = [];
  const warnings: ConvertResult["warnings"] = [];
  const byId = new Map<string, ShapeNode>();
  const byLabel = new Map<string, ShapeNode[]>();
  const elemId = new Map<ShapeNode, string>();
  let counter = 0;

  const common = (style: Style) => ({
    roughness: opts.sketchy ? 1 : 0,
    strokeColor: color(STROKE, style.stroke) ?? "#1e1e1e",
    strokeStyle: style.dashed ? "dashed" : style.dotted ? "dotted" : "solid",
    strokeWidth: 2,
  });
  const fontFamily = opts.sketchy ? 5 : 6; // Excalifont : Nunito

  const fitFont = (text: string, maxW: number, byHeight: number) => {
    const longest = Math.max(...text.split("\n").map((l) => l.length), 1);
    const byWidth = maxW > 0 ? maxW / (longest * (opts.sketchy ? 0.5 : 0.48)) : Infinity;
    return Math.max(8, Math.min(120, byHeight, byWidth));
  };

  const visit = (n: BoardNode) => {
    const b = boxes.get(n)!;
    if (n.type === "table") {
      const size = Number(n.style.size) || 16;
      const pitch = n.rows.length > 1 ? (b.h - size * LINE_HEIGHT) / (n.rows.length - 1) : 0;
      const cols = n.cols.length ? n.cols : n.rows[0]?.map((_, c) => b.x + (c * b.w) / n.rows[0].length) ?? [];
      n.rows.forEach((row, r) => {
        row.forEach((cell, c) => {
          if (!cell) return;
          const right = c + 1 < cols.length ? cols[c + 1] - 8 : b.x + b.w;
          skeletons.push({
            type: "text", x: cols[c] ?? b.x, y: b.y + r * pitch, text: cell,
            fontSize: fitFont(cell, right - (cols[c] ?? b.x), size), fontFamily,
            strokeColor: "#1e1e1e",
          });
        });
      });
      return;
    }
    if (n.type === "text") {
      const lines = n.text.split("\n");
      // Excalidraw's fonts are wider than most screen fonts: also fit the width.
      const fontSize = fitFont(n.text, n.geom?.w ?? 0, b.h / (lines.length * LINE_HEIGHT));
      if (n.style.icon) {
        // [icon]: a small symbol just left of the text
        const s = Math.min(b.h / lines.length, 28) * 0.8;
        skeletons.push({
          type: "rectangle", x: b.x - s - 6, y: b.y + (b.h / lines.length - s) / 2, width: s, height: s,
          roughness: opts.sketchy ? 1 : 0, strokeColor: "#868e96", strokeWidth: 1,
          backgroundColor: "#e9ecef", fillStyle: "solid", roundness: { type: 3 },
        });
      }
      skeletons.push({
        type: "text", x: b.x, y: b.y, text: n.text, fontSize, fontFamily,
        strokeColor: color(STROKE, n.style.color) ?? "#1e1e1e",
      });
      return;
    }
    const id = n.id ?? `shape-${++counter}`;
    if (n.id) {
      if (byId.has(n.id)) warnings.push({ line: n.line, message: `duplicate id "${n.id}"` });
      byId.set(n.id, n);
    }
    if (n.label) byLabel.set(n.label, [...(byLabel.get(n.label) ?? []), n]);
    elemId.set(n, id);
    const fill = color(FILL, n.style.fill);
    const isContainer = n.children.length > 0;
    if (n.kind === "group") {
      // an invisible layout column: just its children
      n.children.forEach(visit);
      return;
    }
    if (n.kind === "icons") {
      const k = Math.max(1, n.count ?? 1);
      const s = Math.min(b.h, b.w / k);
      const step = k > 1 ? (b.w - s) / (k - 1) : 0;
      for (let i = 0; i < k; i++) {
        skeletons.push({
          type: "rectangle", x: b.x + i * step, y: b.y + (b.h - s) / 2, width: s, height: s,
          roughness: opts.sketchy ? 1 : 0, strokeColor: "#868e96", strokeWidth: 1,
          backgroundColor: "#e9ecef", fillStyle: "solid", roundness: { type: 3 },
        });
      }
      return;
    }
    if (n.kind === "icon" || n.kind === "image") {
      // placeholders: a soft rounded square for icons, a crossed box for images
      const icon = n.kind === "icon";
      skeletons.push({
        type: "rectangle", id, x: b.x, y: b.y, width: b.w, height: b.h,
        roughness: opts.sketchy ? 1 : 0, strokeColor: "#868e96", strokeWidth: 1,
        backgroundColor: icon ? "#e9ecef" : "#f1f3f5", fillStyle: icon ? "solid" : "hachure",
        roundness: icon ? { type: 3 } : null,
      });
      if (!icon) {
        for (const [x0, y0, x1, y1] of [[b.x, b.y, b.x + b.w, b.y + b.h], [b.x + b.w, b.y, b.x, b.y + b.h]]) {
          skeletons.push({
            type: "line", x: x0, y: y0, points: [[0, 0], [x1 - x0, y1 - y0]],
            strokeColor: "#ced4da", strokeWidth: 1, roughness: opts.sketchy ? 1 : 0,
          });
        }
      }
      n.children.forEach(visit);
      return;
    }
    skeletons.push({
      type: n.kind === "rect" ? "rectangle" : n.kind,
      id, x: b.x, y: b.y, width: b.w, height: b.h,
      ...common(n.style),
      backgroundColor: fill ?? "transparent",
      fillStyle: "solid",
      label: n.label
        ? {
            text: n.label, fontFamily,
            fontSize: isContainer
              ? LABEL_SIZE
              : fitFont(n.label, b.w - 16, Math.min(LABEL_SIZE, (b.h * 0.7) / (n.label.split("\n").length * LINE_HEIGHT))),
            textAlign: isContainer ? "left" : "center",
            verticalAlign: isContainer ? "top" : "middle",
          }
        : undefined,
    });
    n.children.forEach(visit);
  };
  board.items.forEach(visit);

  const resolve = (e: EdgeNode, ref: EdgeNode["src"]): ShapeNode | undefined => {
    if ("id" in ref) {
      const s = byId.get(ref.id);
      if (!s) warnings.push({ line: e.line, message: `no shape with id "${ref.id}"` });
      return s;
    }
    const all = byLabel.get(ref.label);
    if (!all) warnings.push({ line: e.line, message: `no shape labelled "${ref.label}"` });
    else if (all.length > 1) warnings.push({ line: e.line, message: `"${ref.label}" is ambiguous — give the shapes ids` });
    return all?.[0];
  };

  for (const e of board.edges) {
    const a = resolve(e, e.src);
    const z = resolve(e, e.dst);
    if (!a || !z) continue;
    const ba = boxes.get(a)!, bz = boxes.get(z)!;
    const dx = bz.x + bz.w / 2 - (ba.x + ba.w / 2);
    const dy = bz.y + bz.h / 2 - (ba.y + ba.h / 2);
    if (!dx && !dy) continue;
    const [px, py] = borderPoint(a.kind, ba, dx, dy);
    const [qx, qy] = borderPoint(z.kind, bz, -dx, -dy);
    skeletons.push({
      type: "arrow", x: px, y: py,
      points: [[0, 0], [qx - px, qy - py]],
      width: Math.abs(qx - px), height: Math.abs(qy - py),
      ...common(e.style),
      start: { id: elemId.get(a) },
      end: { id: elemId.get(z) },
      startArrowhead: e.kind === "<-" || e.kind === "<->" ? "arrow" : null,
      endArrowhead: e.kind === "->" || e.kind === "<->" ? "arrow" : null,
      label: e.label ? { text: e.label, fontSize: 16, fontFamily } : undefined,
    });
  }

  return { skeletons, warnings };
}

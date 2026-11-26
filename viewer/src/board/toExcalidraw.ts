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

  const visit = (n: BoardNode) => {
    const b = boxes.get(n)!;
    if (n.type === "text") {
      const lines = n.text.split("\n").length;
      const fontSize = Math.max(8, Math.min(120, b.h / (lines * LINE_HEIGHT)));
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
    skeletons.push({
      type: n.kind === "rect" ? "rectangle" : n.kind,
      id, x: b.x, y: b.y, width: b.w, height: b.h,
      ...common(n.style),
      backgroundColor: fill ?? "transparent",
      fillStyle: "solid",
      label: n.label
        ? {
            text: n.label, fontSize: LABEL_SIZE, fontFamily,
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

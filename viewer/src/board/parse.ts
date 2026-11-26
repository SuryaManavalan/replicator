// Parser for the v0 board syntax (see syntax/SPEC.md).

export type ShapeKind = "rect" | "ellipse" | "diamond" | "image" | "icon";
export type EdgeKind = "->" | "<-" | "<->" | "--";
export type Style = Record<string, string | true>;
export interface Geom { x: number; y: number; w?: number; h?: number }

export interface ShapeNode {
  type: "shape";
  kind: ShapeKind;
  id?: string;
  label?: string;
  geom?: Geom;
  style: Style;
  children: BoardNode[];
  line: number;
}

export interface TextNode {
  type: "text";
  text: string;
  geom?: Geom;
  style: Style;
  line: number;
}

/** Text aligned in rows and columns; cols are absolute column x positions. */
export interface TableNode {
  type: "table";
  geom?: Geom;
  cols: number[];
  rows: string[][];
  style: Style;
  line: number;
}

export type BoardNode = ShapeNode | TextNode | TableNode;
export type Ref = { id: string } | { label: string };

export interface EdgeNode {
  src: Ref;
  dst: Ref;
  kind: EdgeKind;
  label?: string;
  style: Style;
  line: number;
}

export interface ParseError { line: number; message: string }

export interface Board {
  w?: number;
  h?: number;
  items: BoardNode[];
  edges: EdgeNode[];
  errors: ParseError[];
}

const SHAPES = new Set(["rect", "ellipse", "diamond", "image", "icon"]);

type Tok =
  | { t: "str"; v: string }
  | { t: "pos"; x: number; y: number }
  | { t: "size"; w: number; h: number }
  | { t: "style"; v: Style }
  | { t: "arrow"; v: EdgeKind }
  | { t: "colon" }
  | { t: "pipe" }
  | { t: "nums"; v: number[] }
  | { t: "word"; v: string };

const TOKEN = new RegExp(
  [
    String.raw`"(?:[^"\\]|\\.)*"`, // string
    String.raw`@-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?`, // @x,y
    String.raw`\d+(?:\.\d+)?x\d+(?:\.\d+)?`, // WxH
    String.raw`-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?)*`, // 12,40,80 (number list)
    String.raw`\|`,
    String.raw`\[[^\]]*\]`, // [style]
    String.raw`<->|->|<-|--`, // arrows
    String.raw`:`,
    String.raw`[A-Za-z_][\w-]*`, // word
    String.raw`#.*$`, // comment
    String.raw`\S+`, // anything else -> error
  ].join("|"),
  "g",
);

function unquote(s: string): string {
  return s.slice(1, -1).replace(/\\(.)/g, (_, c) => (c === "n" ? "\n" : c));
}

function parseStyle(body: string): Style {
  const style: Style = {};
  for (const part of body.split(/\s+/).filter(Boolean)) {
    const eq = part.indexOf("=");
    if (eq < 0) style[part] = true;
    else style[part.slice(0, eq)] = part.slice(eq + 1);
  }
  return style;
}

function tokenize(src: string): Tok[] | string {
  const toks: Tok[] = [];
  for (const m of src.matchAll(TOKEN)) {
    const s = m[0];
    if (s.startsWith("#")) break;
    if (s.startsWith('"')) toks.push({ t: "str", v: unquote(s) });
    else if (s.startsWith("@")) {
      const [x, y] = s.slice(1).split(",").map(Number);
      toks.push({ t: "pos", x, y });
    } else if (/^\d/.test(s) && s.includes("x")) {
      const [w, h] = s.split("x").map(Number);
      toks.push({ t: "size", w, h });
    } else if (s.startsWith("[")) toks.push({ t: "style", v: parseStyle(s.slice(1, -1)) });
    else if (s === "->" || s === "<-" || s === "<->" || s === "--") toks.push({ t: "arrow", v: s });
    else if (s === ":") toks.push({ t: "colon" });
    else if (s === "|") toks.push({ t: "pipe" });
    else if (/^-?\d/.test(s)) toks.push({ t: "nums", v: s.split(",").map(Number) });
    else if (/^[A-Za-z_]/.test(s)) toks.push({ t: "word", v: s });
    else return `unexpected "${s}"`;
  }
  return toks;
}

/** Consume trailing `@x,y [WxH] [style]` from toks starting at i. */
function tail(toks: Tok[], i: number): { geom?: Geom; style: Style; rest: Tok[] } {
  let geom: Geom | undefined;
  let style: Style = {};
  const rest: Tok[] = [];
  for (; i < toks.length; i++) {
    const k = toks[i];
    if (k.t === "pos") geom = { x: k.x, y: k.y };
    else if (k.t === "size" && geom) Object.assign(geom, { w: k.w, h: k.h });
    else if (k.t === "style") style = { ...style, ...k.v };
    else rest.push(k);
  }
  return { geom, style, rest };
}

function asRef(k: Tok | undefined): Ref | null {
  if (!k) return null;
  if (k.t === "str") return { label: k.v };
  if (k.t === "word") return { id: k.v };
  return null;
}

export function parse(src: string): Board {
  const board: Board = { items: [], edges: [], errors: [] };
  // stack[d] = node at depth d that children at depth d+1 attach to
  const stack: BoardNode[] = [];

  src.split("\n").forEach((raw, idx) => {
    const line = idx + 1;
    const err = (message: string) => board.errors.push({ line, message });
    const body = raw.trimStart();
    if (!body || body.startsWith("#")) return;
    const indent = raw.length - body.length;
    const depth = Math.floor(indent / 2);

    const toks = tokenize(body);
    if (typeof toks === "string") return err(toks);
    if (!toks.length) return;

    const first = toks[0];
    stack.length = Math.min(stack.length, depth);
    const parent = depth > 0 ? stack[depth - 1] : undefined;

    // table row: "a" | "b" | "c"
    if (parent?.type === "table") {
      const cells: string[] = [];
      let expectCell = true;
      for (const k of toks) {
        if (k.t === "str" && expectCell) { cells.push(k.v); expectCell = false; }
        else if (k.t === "pipe") { if (expectCell) cells.push(""); expectCell = true; }
        else return err('table rows look like: "a" | "b" | "c"');
      }
      parent.rows.push(cells);
      return;
    }

    // board WxH
    if (first.t === "word" && first.v === "board") {
      const size = toks.find((k) => k.t === "size");
      if (size && size.t === "size") Object.assign(board, { w: size.w, h: size.h });
      return;
    }

    // edge: REF ARROW REF [: "label"] [style]
    if (toks[1]?.t === "arrow") {
      const src = asRef(first);
      const dst = asRef(toks[2]);
      if (!src || !dst) return err("edge needs a quoted label or id on both sides");
      let label: string | undefined;
      let i = 3;
      if (toks[i]?.t === "colon") {
        const l = toks[i + 1];
        if (l?.t !== "str") return err('edge label must be quoted, e.g. : "label"');
        label = l.v;
        i += 2;
      }
      const { style, rest } = tail(toks, i);
      if (rest.length) err("unexpected tokens after edge");
      board.edges.push({ src, dst, kind: (toks[1] as { v: EdgeKind }).v, label, style, line });
      return;
    }

    let node: BoardNode;
    if (first.t === "word" && first.v === "table") {
      let cols: number[] = [];
      const rest: Tok[] = [];
      for (let i = 1; i < toks.length; i++) {
        const k = toks[i];
        if (k.t === "word" && k.v === "cols" && toks[i + 1]?.t === "nums") {
          cols = (toks[++i] as { v: number[] }).v;
        } else rest.push(k);
      }
      const { geom, style, rest: extra } = tail(rest, 0);
      if (extra.length) err("unexpected tokens after table");
      node = { type: "table", geom, cols, rows: [], style, line };
    } else if (first.t === "word" && SHAPES.has(first.v)) {
      let i = 1;
      let id: string | undefined;
      let label: string | undefined;
      const k1 = toks[i];
      if (k1?.t === "word") { id = k1.v; i++; }
      const k2 = toks[i];
      if (k2?.t === "str") { label = k2.v; i++; }
      const { geom, style, rest } = tail(toks, i);
      if (rest.length) err("unexpected tokens after shape");
      node = { type: "shape", kind: first.v as ShapeKind, id, label, geom, style, children: [], line };
    } else if (first.t === "str") {
      const { geom, style, rest } = tail(toks, 1);
      if (rest.length) err("unexpected tokens after text");
      node = { type: "text", text: first.v, geom, style, line };
    } else {
      return err(`unknown statement "${body.split(/\s/)[0]}"`);
    }

    // attach by indentation
    if (depth > 0 && !parent) {
      err("indented, but nothing above it to sit inside");
      board.items.push(node);
    } else if (parent && parent.type !== "shape") {
      err(`${parent.type} can't contain shapes or text`);
      board.items.push(node);
    } else if (parent) {
      parent.children.push(node);
    } else {
      board.items.push(node);
    }
    stack[depth] = node;
  });

  return board;
}

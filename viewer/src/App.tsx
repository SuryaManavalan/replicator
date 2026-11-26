import { useEffect, useMemo, useState } from "react";
import { Excalidraw, convertToExcalidrawElements } from "@excalidraw/excalidraw";
import type { ExcalidrawImperativeAPI } from "@excalidraw/excalidraw/types";
import "@excalidraw/excalidraw/index.css";
import { parse } from "./board/parse";
import { toSkeletons } from "./board/toExcalidraw";

// Samples: samples/**/NAME.png with optional NAME.board (ground truth) and
// NAME.pred.board (converter output).
const boards = import.meta.glob("../../samples/**/*.board", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
const images = import.meta.glob("../../samples/**/*.png", { query: "?url", import: "default", eager: true }) as Record<string, string>;

type Version = "converted" | "truth";
interface Sample { name: string; image?: string; versions: Partial<Record<Version, string>> }

const samples: Sample[] = (() => {
  const byName = new Map<string, Sample>();
  const get = (name: string) => {
    if (!byName.has(name)) byName.set(name, { name, versions: {} });
    return byName.get(name)!;
  };
  const strip = (p: string) => p.replace("../../samples/", "");
  for (const [p, url] of Object.entries(images)) get(strip(p).replace(/\.png$/, "")).image = url;
  for (const [p, text] of Object.entries(boards)) {
    const n = strip(p);
    if (n.endsWith(".pred.board")) get(n.replace(/\.pred\.board$/, "")).versions.converted = text;
    else get(n.replace(/\.board$/, "")).versions.truth = text;
  }
  return [...byName.values()].filter((s) => Object.keys(s.versions).length).sort((a, b) => a.name.localeCompare(b.name));
})();

const firstVersion = (s?: Sample): Version => (s?.versions.converted !== undefined ? "converted" : "truth");

export default function App() {
  const [index, setIndex] = useState(0);
  const [version, setVersion] = useState<Version>(firstVersion(samples[0]));
  const [text, setText] = useState(samples[0]?.versions[firstVersion(samples[0])] ?? "board 1280x800\n");
  const [sketchy, setSketchy] = useState(true);
  const [api, setApi] = useState<ExcalidrawImperativeAPI | null>(null);
  // Excalidraw measures text when elements are created; re-create them once
  // its fonts finish loading, or widths are computed with a fallback font.
  const [fontEpoch, setFontEpoch] = useState(0);
  useEffect(() => {
    const bump = () => setFontEpoch((n) => n + 1);
    document.fonts.addEventListener("loadingdone", bump);
    return () => document.fonts.removeEventListener("loadingdone", bump);
  }, []);
  const sample = samples[index];

  const { board, skeletons, problems } = useMemo(() => {
    const board = parse(text);
    const { skeletons, warnings } = toSkeletons(board, { sketchy });
    return { board, skeletons, problems: [...board.errors, ...warnings].sort((a, b) => a.line - b.line) };
  }, [text, sketchy]);

  useEffect(() => {
    if (!api) return;
    let elements;
    try {
      elements = convertToExcalidrawElements(skeletons, { regenerateIds: false });
    } catch (e) {
      console.error(e);
      return;
    }
    api.updateScene({ elements });
    api.scrollToContent(elements, { fitToContent: true });
  }, [api, skeletons, fontEpoch]);

  const pick = (i: number, v?: Version) => {
    const s = samples[i];
    const ver = v && s.versions[v] !== undefined ? v : firstVersion(s);
    setIndex(i);
    setVersion(ver);
    setText(s.versions[ver] ?? "");
  };

  return (
    <div className="app">
      <header>
        <strong>replicator</strong>
        <select value={index} onChange={(e) => pick(Number(e.target.value))}>
          {samples.map((s, i) => (
            <option key={s.name} value={i}>{s.name}</option>
          ))}
        </select>
        <div className="seg">
          {(["converted", "truth"] as Version[]).map((v) => (
            <button key={v} className={v === version ? "on" : ""} disabled={sample?.versions[v] === undefined}
              onClick={() => pick(index, v)}>
              {v === "truth" ? "expected" : v}
            </button>
          ))}
        </div>
        <label>
          <input type="checkbox" checked={sketchy} onChange={(e) => setSketchy(e.target.checked)} /> sketchy
        </label>
        <span className="stats">
          {board.w && board.h ? `${board.w}×${board.h} · ` : ""}
          {skeletons.length} elements · {text.length} chars
        </span>
      </header>
      <main>
        <section className="left">
          <div className="source">
            {sample?.image ? <img src={sample.image} alt={sample.name} /> : <p className="empty">no source image</p>}
          </div>
          <div className="editor">
            <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
            {problems.length > 0 && (
              <ul className="problems">
                {problems.map((p, i) => (
                  <li key={i}>line {p.line}: {p.message}</li>
                ))}
              </ul>
            )}
          </div>
        </section>
        <section className="canvas">
          <Excalidraw excalidrawAPI={setApi} viewModeEnabled />
        </section>
      </main>
    </div>
  );
}

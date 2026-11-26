import { useEffect, useMemo, useState } from "react";
import { Excalidraw, convertToExcalidrawElements } from "@excalidraw/excalidraw";
import type { ExcalidrawImperativeAPI } from "@excalidraw/excalidraw/types";
import "@excalidraw/excalidraw/index.css";
import { parse } from "./board/parse";
import { toSkeletons } from "./board/toExcalidraw";

// Every samples/**/NAME.board, paired with NAME.png when present.
const boards = import.meta.glob("../../samples/**/*.board", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
const images = import.meta.glob("../../samples/**/*.png", { query: "?url", import: "default", eager: true }) as Record<string, string>;

const samples = Object.keys(boards)
  .sort()
  .map((path) => ({
    name: path.replace("../../samples/", "").replace(/\.board$/, ""),
    board: boards[path],
    image: images[path.replace(/\.board$/, ".png")],
  }));

export default function App() {
  const [index, setIndex] = useState(0);
  const [text, setText] = useState(samples[0]?.board ?? "board 1280x800\n");
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

  const pick = (i: number) => {
    setIndex(i);
    setText(samples[i].board);
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

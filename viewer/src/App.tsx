import { useEffect, useMemo, useRef, useState } from "react";
import { Excalidraw, convertToExcalidrawElements } from "@excalidraw/excalidraw";
import type { ExcalidrawImperativeAPI } from "@excalidraw/excalidraw/types";
import "@excalidraw/excalidraw/index.css";
import { parse } from "./board/parse";
import { toSkeletons } from "./board/toExcalidraw";

// Still samples: samples/**/NAME.png with optional NAME.board (ground truth)
// and NAME.pred.board (converter output).
const boards = import.meta.glob(["../../samples/**/*.board", "!../../samples/video/*/**"], { query: "?raw", import: "default", eager: true }) as Record<string, string>;
const images = import.meta.glob(["../../samples/**/*.png", "!../../samples/video/*/**"], { query: "?url", import: "default", eager: true }) as Record<string, string>;

// Video samples: samples/video/NAME/{frames/NNNN.png, boards/NNNN.board, steps.json, events.log}
// written by `python -m replicator.video`.
const vFrames = import.meta.glob("../../samples/video/*/frames/*.png", { query: "?url", import: "default", eager: true }) as Record<string, string>;
const vBoards = import.meta.glob("../../samples/video/*/boards/*.board", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
const vSteps = import.meta.glob("../../samples/video/*/steps.json", { import: "default", eager: true }) as Record<string, Step[]>;
const vEvents = import.meta.glob("../../samples/video/*/events.log", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

type Version = "converted" | "truth";
interface Step { t: number; action: string; frame?: number; ms_total?: number; ocr?: string }
interface Frame { t: number; image: string; board: string; action: string; ms?: number }
interface Event { t: number; head: string; body: string }
interface Sample {
  name: string;
  image?: string;
  versions: Partial<Record<Version, string>>;
  video?: { frames: Frame[]; events: Event[] };
}

const parseTime = (stamp: string) => {
  const [m, s] = stamp.slice(1).split(":");
  return Number(m) * 60 + Number(s);
};

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
  const stills = [...byName.values()].filter((s) => Object.keys(s.versions).length);

  const videos: Sample[] = Object.entries(vSteps).map(([p, steps]) => {
    const dir = p.replace(/steps\.json$/, "");
    const name = strip(dir).replace(/\/$/, "");
    const pad = (n: number) => String(n).padStart(4, "0");
    const frames: Frame[] = [];
    let current: Frame | undefined;
    for (const st of steps) {
      if (st.frame) {
        current = {
          t: st.t, action: st.action, ms: st.ms_total,
          image: vFrames[`${dir}frames/${pad(st.frame)}.png`],
          board: vBoards[`${dir}boards/${pad(st.frame)}.board`] ?? "",
        };
        frames.push(current);
      }
    }
    const events: Event[] = (vEvents[`${dir}events.log`] ?? "")
      .split(/\n(?=@\d)/)
      .filter((e) => e.trim())
      .map((e) => {
        const [first, ...rest] = e.split("\n");
        return { t: parseTime(first.split(" ")[0]), head: first, body: rest.join("\n").trim() };
      });
    return { name, versions: { converted: frames[0]?.board ?? "" }, image: frames[0]?.image, video: { frames, events } };
  });

  return [...videos, ...stills].sort((a, b) => a.name.localeCompare(b.name));
})();

const firstVersion = (s?: Sample): Version => (s?.versions.converted !== undefined ? "converted" : "truth");

export default function App() {
  const [index, setIndex] = useState(0);
  const [version, setVersion] = useState<Version>(firstVersion(samples[0]));
  const [text, setText] = useState(samples[0]?.versions[firstVersion(samples[0])] ?? "board 1280x800\n");
  const [sketchy, setSketchy] = useState(true);
  const [api, setApi] = useState<ExcalidrawImperativeAPI | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const fitNext = useRef(true);
  // Excalidraw measures text when elements are created; re-create them once
  // its fonts finish loading, or widths are computed with a fallback font.
  const [fontEpoch, setFontEpoch] = useState(0);
  useEffect(() => {
    const bump = () => setFontEpoch((n) => n + 1);
    document.fonts.addEventListener("loadingdone", bump);
    return () => document.fonts.removeEventListener("loadingdone", bump);
  }, []);
  const sample = samples[index];
  const video = sample?.video;
  const frame = video?.frames[frameIdx];

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
    if (fitNext.current) {  // keep the camera still while a video plays
      api.scrollToContent(elements, { fitToContent: true });
      fitNext.current = false;
    }
  }, [api, skeletons, fontEpoch]);

  // video playback: one processed frame per second
  useEffect(() => {
    if (!playing || !video) return;
    const id = setInterval(() => {
      setFrameIdx((i) => {
        if (i + 1 >= video.frames.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [playing, video]);

  useEffect(() => {
    if (frame) setText(frame.board);
  }, [frame]);

  const pick = (i: number, v?: Version) => {
    const s = samples[i];
    const ver = v && s.versions[v] !== undefined ? v : firstVersion(s);
    fitNext.current = true;
    setIndex(i);
    setVersion(ver);
    setFrameIdx(0);
    setPlaying(false);
    setText(s.video ? s.video.frames[0]?.board ?? "" : s.versions[ver] ?? "");
  };

  const shownEvents = video ? video.events.filter((e) => frame && e.t <= frame.t + 1e-6) : [];
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [shownEvents.length]);

  return (
    <div className="app">
      <header>
        <strong>replicator</strong>
        <select value={index} onChange={(e) => pick(Number(e.target.value))}>
          {samples.map((s, i) => (
            <option key={s.name} value={i}>{s.video ? `▶ ${s.name}` : s.name}</option>
          ))}
        </select>
        {video ? (
          <div className="player">
            <button onClick={() => { if (frameIdx + 1 >= video.frames.length) setFrameIdx(0); setPlaying(!playing); }}>
              {playing ? "❚❚" : "▶"}
            </button>
            <input type="range" min={0} max={Math.max(video.frames.length - 1, 0)} value={frameIdx}
              onChange={(e) => { setPlaying(false); setFrameIdx(Number(e.target.value)); }} />
            <span className="time">{frame ? `${frame.t.toFixed(0)}s · ${frame.action}` : ""}</span>
          </div>
        ) : (
          <div className="seg">
            {(["converted", "truth"] as Version[]).map((v) => (
              <button key={v} className={v === version ? "on" : ""} disabled={sample?.versions[v] === undefined}
                onClick={() => pick(index, v)}>
                {v === "truth" ? "expected" : v}
              </button>
            ))}
          </div>
        )}
        <label>
          <input type="checkbox" checked={sketchy} onChange={(e) => setSketchy(e.target.checked)} /> sketchy
        </label>
        <span className="stats">
          {board.w && board.h ? `${board.w}×${board.h} · ` : ""}
          {skeletons.length} elements · {text.length} chars
          {frame?.ms ? ` · ${frame.ms} ms` : ""}
        </span>
      </header>
      <main>
        <section className="left">
          <div className="source">
            {(frame?.image ?? sample?.image) ? (
              <img src={frame?.image ?? sample?.image} alt={sample.name} />
            ) : (
              <p className="empty">no source image</p>
            )}
          </div>
          {video ? (
            <div className="events" ref={logRef}>
              {shownEvents.length === 0 && <p className="empty">no events yet</p>}
              {shownEvents.map((e, i) => (
                <div key={i} className={"event" + (i === shownEvents.length - 1 ? " latest" : "")}>
                  <div className="head">{e.head}</div>
                  <pre>{e.head.includes("KEY") ? `(full snapshot, ${e.body.split("\n").length} lines)` : e.body}</pre>
                </div>
              ))}
            </div>
          ) : (
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
          )}
        </section>
        <section className="canvas">
          <Excalidraw excalidrawAPI={setApi} viewModeEnabled />
        </section>
      </main>
    </div>
  );
}

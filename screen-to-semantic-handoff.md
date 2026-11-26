> **⚠️ DEPRECATED — background context only.** This doc describes the general idea of what we're building. Its schema, pipeline, tech choices and build order are **not** binding decisions; don't treat them as the plan.

# Handoff: Live Semantic Whiteboard / Screen-to-Semantic Agent Context

You are picking up a design effort that has not been built yet. This document has everything decided so far. Read it fully before writing code.

## 1. Goal

**Phase 1:** An agent that helps while the user works on an Excalidraw-style whiteboard. It reads a text representation of the board, including how it changed over time.

**Final goal:** While the user is on a live video call (Microsoft Teams, on macOS), turn the **shared screen** into the same text representation about once a second, and merge it with a **live transcript** of the call. The result is one live stream that the user's existing, already-authorized work agents read as context.

## 2. Core design principle

The representation exists **to give an LLM context**, not to store data for a renderer.

- **Primary content:** the text on screen, quoted exactly.
- **Secondary:** where that text sits relative to other text and inside which container, because placement changes meaning.
- Coordinates matter but come last. They give grounding, not meaning.
- **"Firm in the truth":** anything inside quotes is exactly what was on screen (OCR output). Everything outside quotes (structure, position words) is inferred.
- It must tokenize well and an LLM should understand it immediately without a spec.

## 3. The schema

### Rules
1. **Text first, quoted exactly.**
2. **Layout through nesting:** containers → children, listed in reading order. Position comes mostly from "what is inside what" and "what comes next."
3. **A position word on each container:** `@left`, `@top-right`, `@center`, `@bottom`, `@left-tall`, `@top-wide`, etc., worked out from where its box sits on screen. An optional exact box `@(x,y,w,h)` on a 0–100 scale for precision.
4. **No role labels from a model.** Containers are generic `box`. The title is the container's first or largest line of text. The reading agent works out roles ("that's a sidebar") itself.
5. **`overlay`** marks something drawn on top (a modal or menu over a dimmed screen).
6. **Edges** (whiteboard arrows): `"API" -> "Postgres" : "SQL"`. The text is the identifier. Add an ID only when two pieces of text are identical.
7. **Keyframes + deltas, as in video encoding:**
   - `KEY`: a full snapshot, sent when the scene changes (new app or slide, or the container tree no longer matches).
   - `Δ`: only what changed. `~` changed (written old → new), `+` added, `-` removed.
   - Seconds where nothing changed produce **nothing**.
8. **Transcript lines** go on the same timeline: `SAY you "..."` / `SAY others "..."`.
9. Consumers should be able to get the **full current state** without replaying deltas. LLMs are bad at keeping track of state across many diffs. The service keeps the current state up to date (see §6).

### Example (combined stream)
```
@00:14:30 KEY  app:"VS Code"
box @top-wide
  "server.ts" | "db.ts"
box @left-tall "EXPLORER"
  "src/"
  "server.ts"
  "db.ts"
box @center "db.ts"
  "const pool = new Pool({"
  "  max: 10,"
  "})"
box @bottom "TERMINAL"
  "$ npm run dev"
  "Server listening on :3000"

@00:14:31 SAY others "can you bump the pool size?"
@00:14:33 Δ   ~ @center "  max: 10," → "  max: 50,"
@00:14:35 SAY you    "done, fifty should be enough"
@00:14:41 Δ   + @bottom "Error: ECONNREFUSED 127.0.0.1:5432"  @(0,92,60,4)
@00:14:43 SAY others "ah, postgres isn't running"
```

### Example with an overlay
```
@00:21:05 KEY  app:"Figma"
box @top-wide
  "Share" | "Present"
box @left-tall "Layers"
  "Header" | "Hero" | "Footer"
box @center "Hero"
  "Ship faster with Claims"
overlay box @center "Share this file"
  "Add emails"
  "Send invite" | "Cancel"
```

### Whiteboard (Excalidraw) use
The same format applies: boxes are `box`, arrows are edges, the canvas is the root. One parser and one agent prompt for both inputs. Converting to and from real `.excalidraw` JSON is optional and comes later. JSON is only the storage or export format, never what the agent reads.

## 4. Vision pipeline: no vision LLM, everything local

Decided: **no vision model** in the loop. Classic computer vision plus local OCR only.

1. **Capture** a frame about once a second (§5).
2. **Change gate:** compare an image fingerprint (or a downscaled pixel diff) with the previous frame. If unchanged, skip. Most seconds of a call are skipped.
3. **Find cut lines:** detect straight edges, borders and dividers (OpenCV Canny + Hough / morphology), plus places where the background colour changes.
4. **Container tree with recursive XY-cut:** split the screen along those lines, then split each piece again until no cut lines are left. Screen UIs are axis-aligned rectangles, so this works well.
5. **OCR:** get text pieces with their boxes. Place each in the **smallest** container that holds it.
6. **Prune with the key rule — a container is kept only if it separates text:**
   - Drop containers with no text.
   - Collapse a container whose text is identical to its only child's (it separates nothing).
   - Keep anything that splits text into groups, **even when small** (e.g., a thin line between two labels).
   - Otherwise prefer the largest containers. The aim is a simple diagram that keeps all the text.
7. **Overlay check:** a sharp new box over a region whose colours darkened → mark it `overlay`.
8. **Diff against the previous tree:** if the containers match, output a `Δ` with only the text changes. If they don't, output a new `KEY`.
9. Add position words and the optional exact box, then output in the schema.

Parameters that will need tuning: edge-detection thresholds, minimum line length for a cut, how much colour change counts as a region boundary, the overlay check, and the change-gate threshold.

## 5. Capture and audio (macOS)

- **ScreenCaptureKit** (Apple's built-in framework) records **just the Teams window and just the Teams app's audio** in one stream. Needs the user's one-time Screen Recording permission.
- **Microphone** is captured separately (AVAudioEngine).
  - Teams audio → `others`; mic → `you`. That gives speaker labels for free.
  - Telling the other speakers apart (pyannote, run locally) is a later addition, not part of the first version.
- **Local OCR options:** Apple Vision (`VNRecognizeTextRequest`; first choice, built in, fast), or PaddleOCR / Tesseract.
- **Local transcription options:** WhisperKit (Core ML, streaming; default choice), Apple's on-device speech recognition (`SpeechAnalyzer`, recent macOS versions), or whisper.cpp.
- Suggested structure: **one small Swift app** for capture, Vision OCR and transcription. It sends frames and text to the diagram logic (Swift or Python with OpenCV; Python is faster to prototype).

## 6. Output: the live state service

A small local service keeps the current board in memory and exposes:
- `GET /state`: the current full snapshot (no replaying needed).
- A live feed (SSE or WebSocket): pushes each `KEY`, `Δ` and `SAY` event as it happens.
- Optionally `GET /history?since=`.

The user's existing authorized work agents subscribe to it.

## 7. Privacy and security constraints (hard requirements)

- **No images or audio leave the machine.** Capture, edge detection, OCR and transcription are all on-device. Only the resulting text stream is exposed, and only to already-authorized agents.
- This is why **Recall.ai** (hosted meeting bot) and **Microsoft's Graph real-time media bots** (C#/.NET on Windows Server in Azure, needs tenant admin consent) were **rejected for now**: both send the media through a cloud service. Keep them in mind only for a future "agent joins the call by itself" mode.
- Tell meeting participants the agent is present. Recording-consent laws (e.g., California's all-party consent rule) and employer policy apply.
- Work laptops may be locked down. Check that Python/OpenCV can be installed and screen recording is allowed. The built-in Apple Vision OCR and speech recognition avoid extra installs.

## 8. Suggested build order

1. **Static prototype (Python):** screenshot in → OpenCV container tree → OCR → pruning → schema text out. Test on real screenshots (IDE, Figma, slides, browser, modal dialogs). Most of the risk is here: rectangle detection and pruning.
2. **Deltas:** feed a sequence of frames; produce `KEY`/`Δ` correctly; add the change gate.
3. **Live state service:** `/state` + SSE.
4. **Mac capture app (Swift):** ScreenCaptureKit at about 1 fps for the Teams window → pipeline.
5. **Audio:** Teams audio + mic → WhisperKit → `SAY` events on the same timeline.
6. **Whiteboard mode:** an Excalidraw import/export adapter for Phase 1.
7. Later: telling speakers apart, an agent joining calls by itself (Recall.ai or Graph bot), and a TypeScript port if needed (the user is strongest in TS/Node/React).

## 9. Open questions

- Can the agent give positions in words only (`below "API"`) and have a compiler work out coordinates? (Relevant to whiteboard editing mode.)
- Should the exact box `@(x,y,w,h)` always be included, or only for deltas and overlays?
- How fine should the position words be (3×3 grid plus `-tall`/`-wide` modifiers proposed)?
- How often to force a keyframe even without a scene change (e.g., every N minutes) so late subscribers can catch up?

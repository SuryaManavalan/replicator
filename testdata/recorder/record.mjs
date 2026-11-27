// Records a varied set of real browser sessions as test videos.
//
// For each scenario: samples/web/NAME.mp4 (the recording), NAME.truth.jsonl
// (visible DOM text + boxes, sampled ~every second, timestamped against the
// video), and NAME.json (what the scenario is).
//
// Usage:  node record.mjs [name-filter]      (needs network for the "site" ones)

import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync, appendFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { homedir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "../../samples/web");
const page = (name, q = "") => pathToFileURL(join(HERE, "pages", name)).href + q;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- helpers

/** A visible mouse cursor (headless recordings have none; real captures do). */
const CURSOR = () => {
  const add = () => {
    if (document.getElementById("__cursor")) return;
    const c = document.createElement("div");
    c.id = "__cursor";
    c.innerHTML = '<svg width="20" height="28" viewBox="0 0 20 28" style="width:20px!important;height:28px!important;display:block"><path d="M1 1 L1 22 L6 17 L10 26 L13 25 L9 16 L16 16 Z" fill="#000" stroke="#fff" stroke-width="1.5"/></svg>';
    Object.assign(c.style, { position: "fixed", left: "-50px", top: "-50px", zIndex: 2147483647, pointerEvents: "none" });
    document.documentElement.appendChild(c);
    addEventListener("mousemove", (e) => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", add); else add();
};

/** Visible text in the viewport, with boxes (CSS px = video px). */
const SNAPSHOT = () => {
  const out = [];
  const vw = innerWidth, vh = innerHeight;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const text = n.textContent.replace(/\s+/g, " ").trim();
    if (!text) continue;
    const el = n.parentElement;
    if (!el || el.closest("script,style,noscript,#__cursor")) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || Number(cs.opacity) === 0) continue;
    // the browser's own check: also catches hidden ancestors, content-visibility
    // (e.g. collapsed sections on mobile Wikipedia) and zero opacity up the tree
    if (el.checkVisibility && !el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true, contentVisibilityAuto: true })) continue;
    const r = document.createRange();
    r.selectNodeContents(n);
    const rects = [...r.getClientRects()].filter((b) => b.width > 1 && b.height > 1 && b.bottom > 0 && b.top < vh && b.right > 0 && b.left < vw);
    if (!rects.length) continue;
    out.push({ text, boxes: rects.map((b) => [Math.round(b.left), Math.round(b.top), Math.round(b.width), Math.round(b.height)]) });
  }
  // Structure: visible tables as cell grids, and code blocks line by line.
  const vis = (r) => r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  const box = (r) => [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)];
  const tables = [];
  for (const t of document.querySelectorAll("table")) {
    const r = t.getBoundingClientRect();
    if (!vis(r) || (t.checkVisibility && !t.checkVisibility({ checkVisibilityCSS: true }))) continue;
    const rows = [];
    for (const tr of t.rows) {
      const rr = tr.getBoundingClientRect();
      if (!vis(rr)) continue;
      rows.push({ box: box(rr), cells: [...tr.cells].map((c) => c.innerText.replace(/\s+/g, " ").trim()) });
    }
    if (rows.length) tables.push({ box: box(r), fully_visible: r.top >= 0 && r.bottom <= vh, rows });
  }
  const code = [];
  for (const pre of document.querySelectorAll("pre")) {
    const r = pre.getBoundingClientRect();
    if (!vis(r)) continue;
    code.push({ box: box(r), fully_visible: r.top >= 0 && r.bottom <= vh, lines: pre.innerText.replace(/\n$/, "").split("\n") });
  }
  return { url: location.href, scrollY: Math.round(scrollY), items: out, tables, code };
};

function makeHelpers(p) {
  let mx = 0, my = 0;
  return {
    wait: (ms) => sleep(ms),
    async move(x, y, steps = 20) { await p.mouse.move(x, y, { steps }); mx = x; my = y; },
    async scroll(dy, steps = 8, pause = 60) {
      for (let i = 0; i < steps; i++) { await p.mouse.wheel(0, dy / steps); await sleep(pause); }
    },
    async jsScroll(dy, steps = 10, pause = 60) {
      for (let i = 0; i < steps; i++) { await p.evaluate((d) => scrollBy(0, d), dy / steps); await sleep(pause); }
    },
    async drag(x0, y0, x1, y1, steps = 25) {
      await p.mouse.move(x0, y0, { steps: 10 }); await p.mouse.down();
      await p.mouse.move(x1, y1, { steps }); await p.mouse.up(); mx = x1; my = y1;
    },
    async type(text, delay = 70) { await p.keyboard.type(text, { delay }); },
    async tryClick(locator, opts = {}) {
      try {
        const l = typeof locator === "string" ? p.locator(locator).first() : locator.first();
        const box = await l.boundingBox({ timeout: 4000 });
        if (box) await p.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 15 });
        await l.click({ timeout: 4000, ...opts });
        return true;
      } catch { return false; }
    },
    async wander(n = 4) {
      const { width, height } = p.viewportSize();
      for (let i = 0; i < n; i++) await this.move(60 + Math.random() * (width - 120), 60 + Math.random() * (height - 120), 25);
    },
  };
}

// ---------------------------------------------------------------- scenarios

const S = [];
const scenario = (name, opts, run) => S.push({ name, ...opts, run });

// --- real sites
scenario("wiki-graph-theory", { kind: "site", url: "https://en.wikipedia.org/wiki/Graph_theory", viewport: [1280, 720],
  about: "Wikipedia article: scroll through prose and figures, follow a link" }, async (p, h) => {
  await h.wait(2000); await h.wander(2);
  for (let i = 0; i < 5; i++) { await h.scroll(450); await h.wait(1500); }
  await h.tryClick(p.locator("#mw-content-text a[href='/wiki/Leonhard_Euler']"));
  await h.wait(2500); await h.scroll(400); await h.wait(2000);
});
scenario("wiki-mobile-kanban", { kind: "site", url: "https://en.m.wikipedia.org/wiki/Kanban", viewport: [390, 844], mobile: true,
  about: "Wikipedia mobile site in a phone-sized viewport, scrolling" }, async (p, h) => {
  await h.wait(2500);
  for (let i = 0; i < 7; i++) { await h.jsScroll(380); await h.wait(1400); }
});
scenario("wiki-dark-table", { kind: "site", url: "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population?vectornightmode=1",
  viewport: [1920, 1080], dark: true, about: "Wikipedia night mode, a large data table, scrolling" }, async (p, h) => {
  await h.wait(2500); await h.wander(2);
  for (let i = 0; i < 6; i++) { await h.scroll(600); await h.wait(1500); }
});
scenario("hackernews", { kind: "site", url: "https://news.ycombinator.com/", viewport: [1440, 900],
  about: "Hacker News front page: scroll, open a comment thread" }, async (p, h) => {
  await h.wait(2000); await h.wander(2); await h.scroll(500); await h.wait(1500);
  await h.tryClick(p.locator("a:has-text('comments')"));
  await h.wait(2500);
  for (let i = 0; i < 4; i++) { await h.scroll(500); await h.wait(1500); }
});
scenario("github-repo", { kind: "site", url: "https://github.com/microsoft/vscode", viewport: [1920, 1080],
  about: "GitHub repository page: file list, README scroll, switch to Issues tab" }, async (p, h) => {
  await h.wait(3000); await h.wander(2);
  for (let i = 0; i < 3; i++) { await h.scroll(600); await h.wait(1500); }
  await p.evaluate(() => scrollTo(0, 0)); await h.wait(1200);
  await h.tryClick(p.locator("#issues-tab"));
  await h.wait(3500); await h.scroll(500); await h.wait(2000);
});
scenario("mdn-dark", { kind: "site", url: "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map",
  viewport: [1366, 768], dark: true, about: "MDN reference page in dark mode with code samples, scrolling" }, async (p, h) => {
  await h.wait(3000); await h.wander(2);
  for (let i = 0; i < 6; i++) { await h.scroll(450); await h.wait(1500); }
});
scenario("python-docs", { kind: "site", url: "https://docs.python.org/3/tutorial/datastructures.html", viewport: [1280, 800],
  about: "Python docs: code blocks and prose, scroll, follow sidebar link" }, async (p, h) => {
  await h.wait(2000);
  for (let i = 0; i < 4; i++) { await h.scroll(450); await h.wait(1500); }
  await h.tryClick(p.locator("a:has-text('Modules')"));
  await h.wait(2500); await h.scroll(400); await h.wait(2000);
});
scenario("excalidraw-draw", { kind: "site", url: "https://excalidraw.com/", viewport: [1600, 900], canvas: true,
  about: "Excalidraw: draw a small architecture diagram (boxes, labels, arrows) — canvas, no DOM text truth" }, async (p, h) => {
  await h.wait(4000);
  await p.mouse.click(800, 650); await h.wait(500); await p.keyboard.press("Escape");
  const box = async (key, x0, y0, x1, y1, label) => {
    await p.keyboard.press(key); await h.drag(x0, y0, x1, y1); await h.wait(400);
    if (label) { await p.mouse.dblclick((x0 + x1) / 2, (y0 + y1) / 2); await h.type(label); await p.keyboard.press("Escape"); }
    await h.wait(700);
  };
  await box("r", 300, 250, 520, 350, "Web App");
  await box("r", 760, 250, 980, 350, "API Server");
  await p.keyboard.press("a"); await h.drag(525, 300, 755, 300); await h.wait(800);
  await box("o", 760, 500, 980, 610, "Postgres");
  await p.keyboard.press("a"); await h.drag(870, 355, 870, 495); await h.wait(800);
  await p.keyboard.press("t"); await p.mouse.click(300, 520); await h.type("Draft architecture"); await p.keyboard.press("Escape");
  await h.wait(2500);
});
scenario("openstreetmap", { kind: "site", url: "https://www.openstreetmap.org/#map=13/51.5072/-0.1276", viewport: [1280, 720], canvas: true,
  about: "OpenStreetMap: pan and zoom a map (labels are in map tiles)" }, async (p, h) => {
  await h.wait(4000);
  await h.drag(800, 400, 600, 300, 30); await h.wait(2000);
  await h.move(700, 380); for (let i = 0; i < 3; i++) { await p.mouse.wheel(0, -300); await h.wait(1200); }
  await h.drag(700, 380, 900, 450, 30); await h.wait(2500);
});
scenario("npr-text", { kind: "site", url: "https://text.npr.org/", viewport: [1024, 768],
  about: "NPR text-only site: headline list, open a story, scroll" }, async (p, h) => {
  await h.wait(2000); await h.scroll(300); await h.wait(1500);
  await h.tryClick(p.locator("ul a").nth(2));
  await h.wait(2500);
  for (let i = 0; i < 3; i++) { await h.scroll(400); await h.wait(1500); }
});
scenario("wiki-search", { kind: "site", url: "https://en.wikipedia.org/wiki/Main_Page", viewport: [1280, 800],
  about: "Search with an autocomplete dropdown, open a result, scroll" }, async (p, h) => {
  await h.wait(2500);
  await h.tryClick(p.locator("input[name=search]:visible"));
  await h.type("Flowchart", 150); await h.wait(2000);
  await p.keyboard.press("Enter"); await h.wait(3000);
  for (let i = 0; i < 3; i++) { await h.scroll(450); await h.wait(1500); }
});
scenario("github-code", { kind: "site", url: "https://github.com/expressjs/express", viewport: [1440, 900],
  about: "GitHub: open a folder, then a source file (code view with line numbers), scroll the code" }, async (p, h) => {
  await h.wait(3000);
  await p.goto("https://github.com/expressjs/express/tree/master/lib"); await h.wait(3000); await h.wander(1);
  await p.goto("https://github.com/expressjs/express/blob/master/lib/application.js"); await h.wait(3500);
  for (let i = 0; i < 4; i++) { await h.scroll(450); await h.wait(1400); }
});
scenario("arxiv-listing", { kind: "site", url: "https://arxiv.org/list/cs.CV/recent", viewport: [1440, 900],
  about: "arXiv listing: dense lists of titles/authors, scrolling" }, async (p, h) => {
  await h.wait(2500);
  for (let i = 0; i < 6; i++) { await h.scroll(500); await h.wait(1400); }
});

// --- local app-like pages (deterministic, no network)
scenario("dashboard-light", { kind: "local", url: page("dashboard.html"), viewport: [1440, 900],
  about: "Ops dashboard: live-updating KPIs, chart redraws, an incident row appears" }, async (p, h) => {
  for (let i = 0; i < 4; i++) { await h.wander(1); await h.wait(3000); }
});
scenario("dashboard-dark", { kind: "local", url: page("dashboard.html", "?theme=dark"), viewport: [1280, 720], dark: true,
  about: "Same dashboard, dark theme, smaller viewport" }, async (p, h) => {
  for (let i = 0; i < 4; i++) { await h.wander(1); await h.wait(3000); }
});
scenario("kanban-drag", { kind: "local", url: page("kanban.html"), viewport: [1366, 768],
  about: "Trello-like board: drag three cards between columns" }, async (p, h) => {
  await h.wait(1500);
  const card = async (text) => { const b = await p.locator(`.card:has-text("${text}")`).first().boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
  const col = async (i) => { const b = await p.locator(".col").nth(i).boundingBox(); return [b.x + b.width / 2, b.y + b.height - 20]; };
  for (const [t, c] of [["SSO with Okta", 1], ["Flaky e2e login test", 2], ["Audit log viewer", 3]]) {
    const [x0, y0] = await card(t); const [x1, y1] = await col(c);
    await h.drag(x0, y0, x1, y1, 35); await h.wait(1800);
  }
  await h.wait(1500);
});
scenario("form-validation", { kind: "local", url: page("form.html"), viewport: [1280, 800],
  about: "Signup form: typing, validation errors, then a confirmation modal" }, async (p, h) => {
  await h.wait(1000);
  await h.tryClick("#first"); await h.type("Ada");
  await h.tryClick("#last"); await h.type("Lovelace");
  await h.tryClick("#email"); await h.type("ada.lovelace");
  await h.tryClick("#pw"); await h.type("abc");
  await h.tryClick("#go"); await h.wait(2000);
  await h.tryClick("#email"); await h.type("@analytical.io");
  await h.tryClick("#pw"); await h.type("defghijk");
  await h.tryClick("#tos");
  await h.tryClick("#go"); await h.wait(3000);
});
scenario("slides", { kind: "local", url: page("slides.html"), viewport: [1920, 1080],
  about: "Slide deck: five slides (title, bullets, metrics, flow diagram, list) with fades" }, async (p, h) => {
  await h.wait(2500);
  for (let i = 0; i < 4; i++) { await p.keyboard.press("ArrowRight"); await h.wait(3000); }
});
scenario("ide-typing", { kind: "local", url: page("ide.html"), viewport: [1600, 900], dark: true,
  about: "VS Code-like editor: code typed line by line, terminal shows a restart and an error" }, async (p, h) => {
  await h.move(700, 400); await h.wait(6000); await h.move(900, 600); await h.wait(9000);
});
scenario("chat", { kind: "local", url: page("chat.html"), viewport: [1280, 720],
  about: "Slack-like channel: messages arrive with typing indicators, user sends one" }, async (p, h) => {
  await h.wait(11000);
  await h.tryClick("#box"); await h.wait(500);
  await p.evaluate(() => sendMessage("Agreed, let's hold the rollout")); await h.wait(3000);
});
scenario("spreadsheet", { kind: "local", url: page("sheet.html"), viewport: [1280, 720],
  about: "Spreadsheet: select and edit cells (totals recompute), then sort by a column" }, async (p, h) => {
  await h.wait(1500);
  await p.evaluate(() => select(1, 1)); await h.wait(1500);
  await p.evaluate(() => edit(1, 1, 45)); await h.wait(1500);
  await p.evaluate(() => edit(3, 3, 520000)); await h.wait(1800);
  await p.evaluate(() => edit(4, 2, 2400000)); await h.wait(1800);
  await p.evaluate(() => sortBy(5)); await h.wait(3000);
});
scenario("diagram-build", { kind: "local", url: page("flow.html"), viewport: [1440, 900],
  about: "Diagram editor: checkout flowchart built node by node with labelled arrows, then a node is moved" }, async (p, h) => {
  const d = (fn) => p.evaluate(fn);
  await h.wait(1000);
  await d(() => diagram.node("cart", "rect", 220, 180, "Cart")); await h.wait(1500);
  await d(() => diagram.node("auth", "diamond", 520, 180, "Logged in?")); await d(() => diagram.edge("cart", "auth")); await h.wait(1500);
  await d(() => diagram.node("pay", "rect", 860, 180, "Payment")); await d(() => diagram.edge("auth", "pay", "yes")); await h.wait(1500);
  await d(() => diagram.node("inv", "rect", 520, 420, "Sign in")); await d(() => diagram.edge("auth", "inv", "no")); await h.wait(1500);
  await d(() => diagram.node("done", "ellipse", 1160, 180, "Order placed")); await d(() => diagram.edge("pay", "done", "ok")); await h.wait(1500);
  await d(() => diagram.node("fail", "rect", 860, 420, "Payment failed")); await d(() => diagram.edge("pay", "fail", "error")); await h.wait(2000);
  await d(() => diagram.moveNode("fail", 1000, 560)); await h.wait(2500);
});
scenario("inbox-dark", { kind: "local", url: page("inbox.html"), viewport: [1440, 900], dark: true,
  about: "Dark email client: hover the list, open messages in the reading pane" }, async (p, h) => {
  await h.wait(1500);
  for (const i of [0, 1, 4]) { await h.tryClick(p.locator(".item").nth(i)); await h.wait(2500); }
  await h.wander(2); await h.wait(1500);
});
scenario("settings", { kind: "local", url: page("settings.html"), viewport: [1280, 800],
  about: "Settings page: toggle switches, open a dropdown menu, switch tabs" }, async (p, h) => {
  await h.wait(1500);
  await h.tryClick(p.locator(".sw").nth(0)); await h.wait(1200);
  await h.tryClick(p.locator(".dd").nth(1)); await h.wait(2000);
  await h.tryClick(p.locator("#tabs div:has-text('Notifications')")); await h.wait(2000);
  await h.tryClick(p.locator(".sw").nth(2)); await h.wait(1200);
  await h.tryClick(p.locator("#tabs div:has-text('Security')")); await h.wait(2000);
  await h.tryClick(p.locator(".sw").nth(0)); await h.wait(2000);
});
scenario("docs-scroll", { kind: "local", url: page("docs.html"), viewport: [1366, 768],
  about: "Docs page with sticky header and sidebars: scroll down through tables/code, then back up" }, async (p, h) => {
  await h.wait(1500);
  for (let i = 0; i < 6; i++) { await h.scroll(350); await h.wait(1300); }
  await h.scroll(-1200, 10); await h.wait(2500);
});

scenario("docs-zoomed", { kind: "local", url: page("docs.html"), viewport: [1280, 720],
  about: "Docs page at 150% browser zoom (much larger text), scrolling" }, async (p, h) => {
  await p.evaluate(() => { document.body.style.zoom = "1.5"; }); await h.wait(1500);
  for (let i = 0; i < 6; i++) { await h.scroll(400); await h.wait(1300); }
});
scenario("multi-app", { kind: "local", url: page("inbox.html"), viewport: [1440, 900],
  about: "Switching between three different apps in one recording (whole-scene changes)" }, async (p, h) => {
  await h.wait(1500); await h.tryClick(p.locator(".item").nth(2)); await h.wait(2500);
  await p.goto(page("sheet.html")); await h.wait(1500); await p.evaluate(() => edit(2, 1, 11)); await h.wait(2500);
  await p.goto(page("kanban.html")); await h.wait(1500); await h.wander(2); await h.wait(2000);
  await p.goto(page("slides.html")); await h.wait(1500); await p.keyboard.press("ArrowRight"); await h.wait(2500);
});
scenario("video-call", { kind: "local", url: page("call.html"), viewport: [1600, 900], dark: true,
  about: "Teams-like call: shared slides in the main stage (advancing), animated video tiles, chat arriving" }, async (p, h) => {
  await h.wait(3000);
  for (let i = 0; i < 4; i++) { await p.evaluate(() => nextSlide()); await h.wander(1); await h.wait(3000); }
});

// ---------------------------------------------------------------- runner

function chromePath() {
  const p = join(homedir(), ".cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell");
  return existsSync(p) ? p : undefined; // else Playwright's own
}

async function record(browser, sc) {
  const [w, hgt] = sc.viewport;
  const tmp = join(OUT, `.tmp-${sc.name}`);
  rmSync(tmp, { recursive: true, force: true });
  const ctx = await browser.newContext({
    viewport: { width: w, height: hgt },
    recordVideo: { dir: tmp, size: { width: w, height: hgt } },
    colorScheme: sc.dark ? "dark" : "light",
    isMobile: !!sc.mobile, hasTouch: !!sc.mobile,
    deviceScaleFactor: 1,
    userAgent: sc.mobile ? "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1" : undefined,
  });
  await ctx.addInitScript(CURSOR);
  const p = await ctx.newPage();
  const t0 = Date.now();
  const truthFile = join(OUT, `${sc.name}.truth.jsonl`);
  writeFileSync(truthFile, "");
  let running = true;
  const sampler = (async () => {
    while (running) {
      const t = (Date.now() - t0) / 1000;
      try {
        const snap = await p.evaluate(SNAPSHOT);
        appendFileSync(truthFile, JSON.stringify({ t: Math.round(t * 100) / 100, ...snap }) + "\n");
      } catch { /* navigating */ }
      await sleep(1000);
    }
  })();
  let ok = true;
  try {
    await p.goto(sc.url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await sc.run(p, makeHelpers(p));
  } catch (e) {
    ok = false;
    console.error(`  ${sc.name}: ${e.message.split("\n")[0]}`);
  }
  running = false;
  await sampler;
  const duration = (Date.now() - t0) / 1000;
  const video = p.video();
  await ctx.close();
  const webm = await video.path();
  const mp4 = join(OUT, `${sc.name}.mp4`);
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", mp4]);
  rmSync(tmp, { recursive: true, force: true });
  writeFileSync(join(OUT, `${sc.name}.json`), JSON.stringify({
    name: sc.name, kind: sc.kind, url: sc.url.startsWith("file:") ? `pages/${sc.url.split("/pages/")[1]}` : sc.url,
    viewport: sc.viewport, theme: sc.dark ? "dark" : "light", mobile: !!sc.mobile, canvas: !!sc.canvas,
    about: sc.about, seconds: Math.round(duration), completed: ok,
  }, null, 2));
  return { name: sc.name, seconds: Math.round(duration), ok };
}

const filter = process.argv[2];
const todo = S.filter((s) => !filter || new RegExp(filter).test(s.name));
mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ executablePath: chromePath() });
const results = [];
const queue = [...todo];
await Promise.all(Array.from({ length: 3 }, async () => {
  while (queue.length) {
    const sc = queue.shift();
    console.log(`recording ${sc.name} ...`);
    const r = await record(browser, sc);
    console.log(`  done ${r.name}: ${r.seconds}s${r.ok ? "" : " (with errors)"}`);
    results.push(r);
  }
}));
await browser.close();
console.log(`\n${results.length} recordings in ${OUT}`);

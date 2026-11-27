import { defineConfig, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";
import { watch } from "node:fs";
import { resolve } from "node:path";

const samples = resolve(__dirname, "../samples");

/** samples/ is outside the viewer root, so Vite doesn't watch it. Watch it
 * ourselves, but only for *finished* outputs (a video's steps.json is written
 * once at the end of its run; still boards), and debounce: a pipeline run
 * writes frames every second and must not trigger a reload storm. The page
 * is told and shows a reload button; it never reloads by itself. */
function watchSamples() {
  return {
    name: "watch-samples",
    configureServer(server: ViteDevServer) {
      let timer: NodeJS.Timeout | undefined;
      watch(samples, { recursive: true }, (_event, file) => {
        if (!file) return;
        // Vite caches every imported file as a module; samples/ is outside its
        // own watcher, so drop the stale copy of anything that changed.
        for (const mod of server.moduleGraph.getModulesByFile(resolve(samples, file)) ?? []) {
          server.moduleGraph.invalidateModule(mod);
        }
        const finished = /steps\.json$/.test(file) || (/\.board$/.test(file) && !/\/boards\//.test(file));
        if (!finished) return;
        clearTimeout(timer);
        timer = setTimeout(() => {
          // Never reload under the user: mark App stale (so a reload picks up
          // new files) and tell the page, which offers a reload button.
          const mod = server.moduleGraph.getModuleById(resolve(__dirname, "src/App.tsx"));
          if (mod) server.moduleGraph.invalidateModule(mod);
          server.ws.send({ type: "custom", event: "samples-changed" });
        }, 3000);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), watchSamples()], // marks the sample list stale; the page picks it up on reload
  define: { "process.env.IS_PREACT": JSON.stringify("false") },
  server: {
    fs: { allow: [".."] }, // samples/ lives outside viewer/
    watch: { ignored: [samples + "/**"] },
  },
});

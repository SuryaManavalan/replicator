import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

const samples = resolve(__dirname, "../samples");

export default defineConfig({
  plugins: [
    react(),
    {
      // samples/ is outside the viewer root, so Vite doesn't watch it by default:
      // without this, new recordings/boards only appear after a server restart.
      name: "watch-samples",
      configureServer(server) {
        server.watcher.add(samples);
      },
    },
  ],
  define: { "process.env.IS_PREACT": JSON.stringify("false") },
  server: { fs: { allow: [".."] } }, // samples/ lives outside viewer/
});

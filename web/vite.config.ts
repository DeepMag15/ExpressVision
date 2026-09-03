import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to `exv serve`, so the console runs same-origin in
// development exactly as it does when served from the built bundle. That keeps
// CORS out of the picture for everything except an explicitly --dev backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});

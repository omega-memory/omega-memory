// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";

// Select adapter based on ASTRO_ADAPTER env var:
// - "cloudflare" (default for CF Workers deployment)
// - "node" (for Docker / self-hosted)
const adapterName = process.env.ASTRO_ADAPTER || "cloudflare";

let adapter;
if (adapterName === "node") {
  const node = (await import("@astrojs/node")).default;
  adapter = node({ mode: "standalone" });
} else {
  const cloudflare = (await import("@astrojs/cloudflare")).default;
  adapter = cloudflare({ platformProxy: { enabled: true } });
}

export default defineConfig({
  output: "server",
  adapter,
  security: {
    checkOrigin: false, // Handled by Caddy reverse proxy
  },
  integrations: [
    react(),
  ],
  vite: {
    css: {
      postcss: "./postcss.config.mjs",
    },
    optimizeDeps: {
      include: ["react", "react-dom", "react-dom/client"],
    },
    ssr: {
      external: ["pg", "postgres", "drizzle-orm", "drizzle-orm/d1", "drizzle-orm/postgres-js", "drizzle-orm/better-sqlite3", "better-sqlite3", "react-force-graph-3d", "3d-force-graph", "three"],
    },
    build: {
      rollupOptions: {
        external: ["better-sqlite3"],
      },
    },
  },
});

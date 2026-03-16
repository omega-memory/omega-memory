// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";

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
  integrations: [
    react(),
    tailwind({ applyBaseStyles: false }),
  ],
  vite: {
    ssr: {
      external: ["pg", "postgres"],
    },
  },
});

import type { APIRoute } from "astro";
import { getDb } from "../../lib/db";
import { schemaVersion } from "../../lib/db/schema";

export const GET: APIRoute = async (context) => {
  const now = new Date().toISOString();

  let dbStatus = "unknown";
  let dbVersion: number | null = null;

  try {
    const db = getDb(context);
    const [row] = await db.select().from(schemaVersion).limit(1);
    dbVersion = row?.version ?? null;
    dbStatus = "ok";
  } catch (err) {
    dbStatus = "error";
  }

  const status = dbStatus === "ok" ? "healthy" : "degraded";

  return new Response(
    JSON.stringify({
      status,
      timestamp: now,
      database: {
        status: dbStatus,
        schemaVersion: dbVersion,
      },
      runtime: typeof globalThis.caches !== "undefined" ? "cloudflare" : "node",
    }),
    {
      status: status === "healthy" ? 200 : 503,
      headers: { "Content-Type": "application/json" },
    }
  );
};

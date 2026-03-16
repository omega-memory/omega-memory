/**
 * Database client factory — returns a Drizzle instance for either D1 or PostgreSQL.
 *
 * On Cloudflare Workers: uses D1 binding from env
 * On Node.js (Docker/self-hosted): uses PostgreSQL via `postgres` driver
 *
 * Usage in Astro endpoints:
 *   const db = getDb(context);
 *   const rows = await db.select().from(memories).where(...);
 */
import { drizzle as drizzleD1 } from "drizzle-orm/d1";
import * as schema from "./schema";

export type Database = ReturnType<typeof createDb>;

/**
 * Get database from Astro context (works in both CF Workers and Node).
 */
export function getDb(context: { locals: Record<string, unknown> }): Database {
  // Check if already cached on locals
  if (context.locals._db) {
    return context.locals._db as Database;
  }

  const db = createDb(context.locals.runtime as RuntimeEnv | undefined);
  context.locals._db = db;
  return db;
}

interface RuntimeEnv {
  env?: {
    DB?: D1Database;
    DATABASE_URL?: string;
  };
}

function createDb(runtime?: RuntimeEnv) {
  // 1. Cloudflare D1 binding
  const d1 = runtime?.env?.DB;
  if (d1) {
    return drizzleD1(d1, { schema });
  }

  // 2. PostgreSQL via DATABASE_URL
  const pgUrl = runtime?.env?.DATABASE_URL ?? process.env.DATABASE_URL;
  if (pgUrl) {
    // Dynamic import to avoid bundling pg in CF Workers
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { drizzle: drizzlePg } = require("drizzle-orm/postgres-js");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const postgres = require("postgres");
    const client = postgres(pgUrl);
    return drizzlePg(client, { schema });
  }

  throw new Error(
    "No database configured. Set D1 binding (CF) or DATABASE_URL (Node)."
  );
}

export { schema };

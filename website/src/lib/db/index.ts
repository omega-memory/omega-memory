/**
 * Database client factory — returns a Drizzle instance.
 *
 * Priority:
 *   1. Cloudflare D1 binding (Workers)
 *   2. PostgreSQL via DATABASE_URL (Docker / self-hosted)
 *   3. Supabase via SUPABASE_DB_URL (managed Postgres)
 *   4. SQLite fallback via OMEGA_DB_PATH or ./omega.db
 *
 * Usage in Astro endpoints:
 *   const db = await getDb(context);
 *   const rows = await db.select().from(memories).where(...);
 */
import { drizzle as drizzleD1 } from "drizzle-orm/d1";
import * as schema from "./schema";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Database = any;

/**
 * Get database from Astro context (works in both CF Workers and Node).
 */
export async function getDb(context: { locals: App.Locals | Record<string, unknown> }): Promise<Database> {
  // Check if already cached on locals
  if (context.locals._db) {
    return context.locals._db as Database;
  }

  const db = await createDb(context.locals.runtime as RuntimeEnv | undefined);
  context.locals._db = db;
  return db;
}

interface RuntimeEnv {
  env?: {
    DB?: D1Database;
    DATABASE_URL?: string;
    SUPABASE_DB_URL?: string;
  };
}

async function createDb(runtime?: RuntimeEnv) {
  // 1. Cloudflare D1 binding
  const d1 = runtime?.env?.DB;
  if (d1) {
    return drizzleD1(d1, { schema });
  }

  // 2. PostgreSQL via DATABASE_URL
  const pgUrl = runtime?.env?.DATABASE_URL ?? process.env.DATABASE_URL;
  if (pgUrl) {
    const { drizzle: drizzlePg } = await import("drizzle-orm/postgres-js");
    const pgMod = await import("postgres");
    const postgres = pgMod.default ?? pgMod;
    const client = postgres(pgUrl);
    return drizzlePg(client, { schema });
  }

  // 3. Supabase via SUPABASE_DB_URL
  const supabaseUrl = runtime?.env?.SUPABASE_DB_URL ?? process.env.SUPABASE_DB_URL;
  if (supabaseUrl) {
    const { drizzle: drizzlePg } = await import("drizzle-orm/postgres-js");
    const pgMod = await import("postgres");
    const postgres = pgMod.default ?? pgMod;
    const client = postgres(supabaseUrl);
    return drizzlePg(client, { schema });
  }

  // 4. SQLite fallback
  const sqlitePath = process.env.OMEGA_DB_PATH ?? "./omega.db";
  try {
    const { drizzle: drizzleSqlite } = await import("drizzle-orm/better-sqlite3");
    const betterSqlite = await import("better-sqlite3");
    const Database = betterSqlite.default ?? betterSqlite;
    const sqlite = new Database(sqlitePath);
    return drizzleSqlite(sqlite, { schema });
  } catch {
    throw new Error(
      "No database configured. Set one of:\n" +
      "  - D1 binding (Cloudflare Workers)\n" +
      "  - DATABASE_URL (PostgreSQL)\n" +
      "  - SUPABASE_DB_URL (Supabase)\n" +
      "  - OMEGA_DB_PATH (SQLite) or install better-sqlite3"
    );
  }
}

export { schema };

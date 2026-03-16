#!/usr/bin/env node
/**
 * Create a superadmin user for the Omega dashboard.
 *
 * Usage:
 *   node scripts/create-admin.mjs --email admin@example.com --password secret123
 *   node scripts/create-admin.mjs  # interactive prompts
 *
 * Env:
 *   DATABASE_URL        — PostgreSQL / Supabase connection string
 *   SUPABASE_DB_URL     — Supabase connection string (alternative)
 *   OMEGA_DB_PATH       — SQLite file path (fallback: ./omega.db)
 *
 * Priority: DATABASE_URL > SUPABASE_DB_URL > D1 (Cloudflare) > SQLite fallback
 */
import { webcrypto } from "node:crypto";
import { parseArgs } from "node:util";
import { createInterface } from "node:readline";
import { existsSync } from "node:fs";

// Polyfill crypto.subtle for Node <20
if (!globalThis.crypto?.subtle) {
  globalThis.crypto = webcrypto;
}

// ── Arg parsing ──────────────────────────────────────────────────────
const { values } = parseArgs({
  options: {
    email: { type: "string", short: "e" },
    password: { type: "string", short: "p" },
    name: { type: "string", short: "n" },
    "database-url": { type: "string", short: "d" },
    "sqlite-path": { type: "string", short: "s" },
  },
  strict: false,
});

// ── Interactive prompt helper ────────────────────────────────────────
function ask(question, hidden = false) {
  return new Promise((resolve) => {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    if (hidden) {
      process.stdout.write(question);
      const stdin = process.stdin;
      const wasRaw = stdin.isRaw;
      if (stdin.setRawMode) stdin.setRawMode(true);
      let input = "";
      const onData = (ch) => {
        const c = ch.toString();
        if (c === "\n" || c === "\r") {
          stdin.removeListener("data", onData);
          if (stdin.setRawMode) stdin.setRawMode(wasRaw);
          process.stdout.write("\n");
          rl.close();
          resolve(input);
        } else if (c === "\u007f" || c === "\b") {
          if (input.length > 0) {
            input = input.slice(0, -1);
            process.stdout.write("\b \b");
          }
        } else {
          input += c;
          process.stdout.write("•");
        }
      };
      stdin.on("data", onData);
    } else {
      rl.question(question, (answer) => {
        rl.close();
        resolve(answer.trim());
      });
    }
  });
}

// ── Crypto (matches AuthEngine.hashPassword exactly) ─────────────────
function bufToHex(buf) {
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function hashPassword(password) {
  const enc = new TextEncoder();
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    "PBKDF2",
    false,
    ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: 100_000, hash: "SHA-256" },
    keyMaterial,
    256
  );
  const hash = bufToHex(bits);
  const saltHex = bufToHex(salt);
  return `pbkdf2:100000:${saltHex}:${hash}`;
}

// ── Database adapters ────────────────────────────────────────────────

const CREATE_TABLE_SQL = `
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    display_name TEXT,
    role TEXT DEFAULT 'user',
    created_at TEXT,
    updated_at TEXT,
    is_active INTEGER DEFAULT 1
  )
`;

/**
 * PostgreSQL / Supabase adapter
 */
async function runPostgres(url, { email, passwordHash, displayName, id, now }) {
  const pgMod = await import("postgres");
  const postgres = pgMod.default ?? pgMod;
  const sql = postgres(url, { max: 1 });

  try {
    await sql.unsafe(CREATE_TABLE_SQL);

    // Add role column if missing (existing tables from before role support)
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user'`.catch(() => {});

    const existing = await sql`SELECT id, email FROM users WHERE email = ${email}`;
    if (existing.length > 0) {
      console.log(`⚠ User ${email} already exists (id: ${existing[0].id})`);
      const update = await ask("Update password and set as superadmin? [y/N]: ");
      if (update.toLowerCase() !== "y") {
        console.log("Aborted.");
        await sql.end();
        process.exit(0);
      }
      await sql`
        UPDATE users SET password_hash = ${passwordHash}, role = 'superadmin', updated_at = ${now}
        WHERE email = ${email}
      `;
      console.log(`✓ Updated ${email} → superadmin`);
    } else {
      await sql`
        INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at, is_active)
        VALUES (${id}, ${email}, ${passwordHash}, ${displayName}, 'superadmin', ${now}, ${now}, 1)
      `;
      console.log(`✓ Created superadmin: ${email} (id: ${id})`);
    }

    await sql.end();
  } catch (err) {
    console.error("✗ PostgreSQL error:", err.message);
    await sql.end();
    process.exit(1);
  }
}

/**
 * SQLite adapter (fallback for local dev / self-hosted without Postgres)
 */
async function runSqlite(dbPath, { email, passwordHash, displayName, id, now }) {
  let Database;
  try {
    const mod = await import("better-sqlite3");
    Database = mod.default ?? mod;
  } catch {
    console.error("✗ better-sqlite3 not installed. Run: npm install better-sqlite3");
    process.exit(1);
  }

  const db = new Database(dbPath);

  try {
    db.exec(CREATE_TABLE_SQL);

    const existing = db.prepare("SELECT id, email FROM users WHERE email = ?").get(email);
    if (existing) {
      console.log(`⚠ User ${email} already exists (id: ${existing.id})`);
      const update = await ask("Update password and set as superadmin? [y/N]: ");
      if (update.toLowerCase() !== "y") {
        console.log("Aborted.");
        db.close();
        process.exit(0);
      }
      db.prepare("UPDATE users SET password_hash = ?, role = 'superadmin', updated_at = ? WHERE email = ?")
        .run(passwordHash, now, email);
      console.log(`✓ Updated ${email} → superadmin`);
    } else {
      db.prepare(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at, is_active) VALUES (?, ?, ?, ?, 'superadmin', ?, ?, 1)"
      ).run(id, email, passwordHash, displayName, now, now);
      console.log(`✓ Created superadmin: ${email} (id: ${id})`);
    }

    db.close();
  } catch (err) {
    console.error("✗ SQLite error:", err.message);
    db.close();
    process.exit(1);
  }
}

// ── Main ─────────────────────────────────────────────────────────────
async function main() {
  const email = values.email || (await ask("Email: "));
  const password = values.password || (await ask("Password: ", true));
  const displayName = values.name || (await ask("Display name (optional): ")) || null;

  if (!email || !password) {
    console.error("✗ Email and password are required.");
    process.exit(1);
  }

  console.log(`\nCreating admin: ${email}`);

  const passwordHash = await hashPassword(password);
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  const payload = { email, passwordHash, displayName, id, now };

  // Resolve database: PostgreSQL / Supabase > SQLite
  const pgUrl = values["database-url"]
    || process.env.DATABASE_URL
    || process.env.SUPABASE_DB_URL;

  const sqlitePath = values["sqlite-path"]
    || process.env.OMEGA_DB_PATH;

  if (pgUrl) {
    const label = process.env.SUPABASE_DB_URL && !process.env.DATABASE_URL ? "Supabase" : "PostgreSQL";
    console.log(`Using ${label}: ${pgUrl.replace(/\/\/.*@/, "//***@")}`);
    await runPostgres(pgUrl, payload);
  } else if (sqlitePath || existsSync("./omega.db")) {
    const path = sqlitePath || "./omega.db";
    console.log(`Using SQLite: ${path}`);
    await runSqlite(path, payload);
  } else {
    console.error(
      "✗ No database configured.\n" +
      "  Set one of:\n" +
      "    DATABASE_URL        — PostgreSQL connection string\n" +
      "    SUPABASE_DB_URL     — Supabase connection string\n" +
      "    OMEGA_DB_PATH       — SQLite file path\n" +
      "    --database-url      — PostgreSQL/Supabase URL (flag)\n" +
      "    --sqlite-path       — SQLite file path (flag)"
    );
    process.exit(1);
  }
}

main();

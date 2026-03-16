import { defineConfig } from "drizzle-kit";

export default defineConfig({
  schema: "./src/lib/db/schema.ts",
  out: "./drizzle",
  dialect: "sqlite",
  // For D1: use wrangler d1 migrations apply
  // For PG: override dialect to "postgresql" and set dbCredentials
  ...(process.env.DATABASE_URL
    ? {
        dialect: "postgresql" as const,
        dbCredentials: { url: process.env.DATABASE_URL },
      }
    : {}),
});

/// <reference types="astro/client" />

declare namespace App {
  interface Locals {
    runtime?: {
      env?: {
        DB?: D1Database;
        DATABASE_URL?: string;
        AUTH_SECRET?: string;
        ANTHROPIC_API_KEY?: string;
        STRIPE_SECRET_KEY?: string;
        STRIPE_WEBHOOK_SECRET?: string;
      };
    };
    _db?: import("./lib/db").Database;
    user?: Record<string, unknown>;
    [key: string]: unknown;
  }
}

// Cloudflare D1 types (minimal)
interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T = unknown>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]>;
  exec(query: string): Promise<D1ExecResult>;
}

interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  first<T = unknown>(colName?: string): Promise<T | null>;
  run<T = unknown>(): Promise<D1Result<T>>;
  all<T = unknown>(): Promise<D1Result<T>>;
  raw<T = unknown>(): Promise<T[]>;
}

interface D1Result<T = unknown> {
  results?: T[];
  success: boolean;
  error?: string;
  meta?: Record<string, unknown>;
}

interface D1ExecResult {
  count: number;
  duration: number;
}

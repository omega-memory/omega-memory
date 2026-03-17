/**
 * Supabase Auth Provider — validates Supabase JWTs and relays OAuth to Supabase Auth.
 *
 * Uses ONLY Web Crypto API (works on Cloudflare Workers).
 * No node:crypto, no jsonwebtoken, no Buffer.
 *
 * Env vars: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_JWT_SECRET, SUPABASE_AUTH_PROVIDERS
 */
import { eq, and } from "drizzle-orm";
import type { AuthProvider, AuthResult, LoginMethod } from "../provider";
import type { Database } from "../../db";
import { AuthEngine } from "../index";
import { users } from "../../db/schema";

// ===================================================================
// JWT helpers (Web Crypto only — no libraries)
// ===================================================================

function base64UrlDecode(str: string): Uint8Array {
  // Replace URL-safe chars with standard base64 chars
  const base64 = str.replace(/-/g, "+").replace(/_/g, "/");
  // Pad to multiple of 4
  const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    const json = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch {
    return null;
  }
}

async function verifySupabaseJwt(
  token: string,
  secret: string
): Promise<Record<string, unknown> | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );

  const data = enc.encode(`${parts[0]}.${parts[1]}`);
  const sigBytes = base64UrlDecode(parts[2]);

  // Copy into a fresh ArrayBuffer to satisfy strict TypeScript BufferSource typing
  const sigBuffer = new ArrayBuffer(sigBytes.byteLength);
  new Uint8Array(sigBuffer).set(sigBytes);

  const valid = await crypto.subtle.verify("HMAC", key, sigBuffer, data);
  if (!valid) return null;

  const payload = decodeJwtPayload(token);
  if (!payload) return null;

  // Check expiration
  if (
    typeof payload.exp === "number" &&
    payload.exp < Date.now() / 1000
  ) {
    return null;
  }

  return payload;
}

// ===================================================================
// Helper: extract JWT from request
// ===================================================================

async function extractToken(request: Request): Promise<string | null> {
  // 1. Authorization: Bearer <token>
  const authHeader = request.headers.get("authorization");
  if (authHeader?.startsWith("Bearer ")) {
    return authHeader.slice(7).trim();
  }

  // 2. JSON body with access_token
  const contentType = request.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      const body = await request.json();
      if (typeof body.access_token === "string" && body.access_token) {
        return body.access_token;
      }
    } catch {
      // ignore parse errors
    }
  }

  // 3. Query param (for callback GET requests)
  const url = new URL(request.url);
  const queryToken = url.searchParams.get("access_token");
  if (queryToken) return queryToken;

  return null;
}

// ===================================================================
// Helper: upsert user from Supabase JWT claims
// ===================================================================

async function upsertUserFromClaims(
  db: Database,
  payload: Record<string, unknown>
): Promise<{ id: string; email: string; displayName: string | null; role: string | null }> {
  const email = (payload.email as string) || "";
  const sub = (payload.sub as string) || "";
  const userMetadata = (payload.user_metadata as Record<string, unknown>) || {};
  const displayName =
    (userMetadata.full_name as string) ||
    (userMetadata.name as string) ||
    (userMetadata.preferred_username as string) ||
    null;

  if (!email) {
    throw new Error("Supabase JWT missing email claim");
  }

  // Check if user exists by email
  const [existing] = await db
    .select()
    .from(users)
    .where(eq(users.email, email))
    .limit(1);

  if (existing) {
    // Update display name if we got one and user doesn't have one
    if (displayName && !existing.displayName) {
      await db
        .update(users)
        .set({ displayName, updatedAt: new Date().toISOString() })
        .where(eq(users.id, existing.id));
    }
    return {
      id: existing.id,
      email: existing.email,
      displayName: existing.displayName ?? displayName,
      role: existing.role ?? null,
    };
  }

  // Create new user — passwordHash is null (OAuth user)
  const id = sub || crypto.randomUUID();
  const now = new Date().toISOString();
  await db.insert(users).values({
    id,
    email,
    passwordHash: null,
    displayName,
    role: "user",
    createdAt: now,
    updatedAt: now,
    isActive: 1,
  });

  return { id, email, displayName, role: "user" };
}

// ===================================================================
// Helper: get env var from runtime or process.env
// ===================================================================

function getEnv(name: string, locals?: Record<string, unknown>): string {
  const runtime = locals?.runtime as { env?: Record<string, string> } | undefined;
  return runtime?.env?.[name] ?? (typeof process !== "undefined" ? (process.env as Record<string, string>)[name] : "") ?? "";
}

// ===================================================================
// SupabaseAuthProvider
// ===================================================================

export class SupabaseAuthProvider implements AuthProvider {
  readonly name = "supabase";

  async authenticate(
    request: Request,
    db: Database,
    secret: string
  ): Promise<AuthResult | null> {
    // We need SUPABASE_JWT_SECRET — try to get it from the request context
    // The caller should pass it as the `secret` param, or we fall back to env
    const jwtSecret = this.getJwtSecret(request);
    if (!jwtSecret) return null;

    const token = await extractToken(request);
    if (!token) return null;

    const payload = await verifySupabaseJwt(token, jwtSecret);
    if (!payload) return null;

    const user = await upsertUserFromClaims(db, payload);

    const auth = new AuthEngine(db, secret);
    const deviceInfo = request.headers.get("user-agent") ?? undefined;
    const sessionToken = await auth.createSession(user.id, deviceInfo);

    return {
      user: {
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        role: user.role,
      },
      sessionToken,
    };
  }

  /**
   * Authenticate with explicit env vars (for use in API routes where we have access to locals).
   */
  async authenticateWithEnv(
    request: Request,
    db: Database,
    authSecret: string,
    jwtSecret: string
  ): Promise<AuthResult | null> {
    const token = await extractToken(request);
    if (!token) return null;

    const payload = await verifySupabaseJwt(token, jwtSecret);
    if (!payload) return null;

    const user = await upsertUserFromClaims(db, payload);

    const auth = new AuthEngine(db, authSecret);
    const deviceInfo = request.headers.get("user-agent") ?? undefined;
    const sessionToken = await auth.createSession(user.id, deviceInfo);

    return {
      user: {
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        role: user.role,
      },
      sessionToken,
    };
  }

  async getAuthorizationUrl(state: string, redirectUri: string): Promise<string> {
    // `state` is used here as the OAuth provider name (e.g., "google", "github")
    // The actual Supabase URL and redirect are constructed from env vars
    // Caller should provide the SUPABASE_URL in the redirectUri format or we parse from env
    throw new Error(
      "Use getAuthorizationUrlWithEnv() instead — requires SUPABASE_URL"
    );
  }

  /**
   * Build Supabase OAuth authorization URL.
   * @param provider OAuth provider name (e.g., "google", "github")
   * @param redirectUri Where Supabase should redirect after auth
   * @param supabaseUrl The Supabase project URL
   */
  getAuthorizationUrlWithEnv(
    provider: string,
    redirectUri: string,
    supabaseUrl: string
  ): string {
    const url = new URL(`${supabaseUrl}/auth/v1/authorize`);
    url.searchParams.set("provider", provider);
    url.searchParams.set("redirect_to", redirectUri);
    return url.toString();
  }

  async handleCallback(
    request: Request,
    db: Database,
    secret: string
  ): Promise<AuthResult | null> {
    // Same as authenticate — the callback just has a JWT to validate
    return this.authenticate(request, db, secret);
  }

  /**
   * Handle callback with explicit env vars.
   */
  async handleCallbackWithEnv(
    request: Request,
    db: Database,
    authSecret: string,
    jwtSecret: string
  ): Promise<AuthResult | null> {
    return this.authenticateWithEnv(request, db, authSecret, jwtSecret);
  }

  getLoginMethods(): LoginMethod[] {
    // Read providers from env — default to empty
    const providersStr =
      typeof process !== "undefined"
        ? process.env.SUPABASE_AUTH_PROVIDERS ?? ""
        : "";
    const providers = providersStr
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    return [{ type: "supabase", providers }];
  }

  /**
   * Get login methods with explicit providers list.
   */
  getLoginMethodsWithProviders(providers: string[]): LoginMethod[] {
    return [{ type: "supabase", providers }];
  }

  // --- Private helpers ---

  private getJwtSecret(request: Request): string | null {
    // Try to extract from process.env as fallback
    if (typeof process !== "undefined" && process.env.SUPABASE_JWT_SECRET) {
      return process.env.SUPABASE_JWT_SECRET;
    }
    return null;
  }
}

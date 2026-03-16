/**
 * Auth engine — Web Crypto API based (works on CF Workers + Node 22+).
 *
 * Replaces the Supabase auth dependency. Supports:
 * - Password auth (scrypt via Web Crypto)
 * - Session tokens (HMAC-SHA256 signed)
 * - API tokens
 * - License key validation
 * - WebAuthn/passkey (via @simplewebauthn)
 */
import { eq, and } from "drizzle-orm";
import { users, userSessions, apiTokens, licenses } from "../db/schema";
import type { Database } from "../db";

const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const PRO_SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
const COOKIE_NAME = "omega_session";
const PRO_COOKIE_NAME = "omega_pro_session";

// ===================================================================
// Crypto helpers (Web Crypto API — works on CF Workers + Node 22+)
// ===================================================================

async function getSigningKey(secret: string): Promise<CryptoKey> {
  const enc = new TextEncoder();
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

async function hmacSign(data: string, secret: string): Promise<string> {
  const key = await getSigningKey(secret);
  const enc = new TextEncoder();
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(data));
  return bufToHex(new Uint8Array(sig));
}

async function hmacVerify(data: string, signature: string, secret: string): Promise<boolean> {
  const expected = await hmacSign(data, secret);
  return timingSafeEqual(expected, signature);
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

function bufToHex(buf: Uint8Array): string {
  return Array.from(buf).map(b => b.toString(16).padStart(2, "0")).join("");
}

async function hashPassword(password: string): Promise<string> {
  // Use PBKDF2 (available in Web Crypto) since scrypt isn't in Web Crypto
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
  const hash = bufToHex(new Uint8Array(bits));
  const saltHex = bufToHex(salt);
  return `pbkdf2:100000:${saltHex}:${hash}`;
}

async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const parts = stored.split(":");
  if (parts[0] !== "pbkdf2" || parts.length !== 4) return false;
  const [, iterations, saltHex, expectedHash] = parts;
  const enc = new TextEncoder();
  const salt = new Uint8Array(saltHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    "PBKDF2",
    false,
    ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: parseInt(iterations), hash: "SHA-256" },
    keyMaterial,
    256
  );
  const hash = bufToHex(new Uint8Array(bits));
  return timingSafeEqual(hash, expectedHash);
}

function generateId(): string {
  return crypto.randomUUID();
}

function generateToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return bufToHex(bytes);
}

async function hashToken(token: string): Promise<string> {
  const enc = new TextEncoder();
  const digest = await crypto.subtle.digest("SHA-256", enc.encode(token));
  return bufToHex(new Uint8Array(digest));
}

// ===================================================================
// Auth Engine
// ===================================================================

export class AuthEngine {
  constructor(
    private db: Database,
    private secret: string
  ) {}

  // --- User management ---

  async createUser(email: string, password: string, displayName?: string) {
    const id = generateId();
    const passwordHashed = await hashPassword(password);
    const now = new Date().toISOString();
    await this.db.insert(users).values({
      id,
      email,
      passwordHash: passwordHashed,
      displayName: displayName ?? null,
      createdAt: now,
      updatedAt: now,
      isActive: 1,
    });
    return { id, email, displayName };
  }

  async authenticateUser(email: string, password: string) {
    const [user] = await this.db
      .select()
      .from(users)
      .where(and(eq(users.email, email), eq(users.isActive, 1)))
      .limit(1);

    if (!user || !user.passwordHash) return null;
    const valid = await verifyPassword(password, user.passwordHash);
    if (!valid) return null;
    return user;
  }

  // --- Session management ---

  async createSession(userId: string, deviceInfo?: string): Promise<string> {
    const token = generateToken();
    const tokenHashed = await hashToken(token);
    const now = new Date();
    const expiresAt = new Date(now.getTime() + SESSION_TTL_MS);

    await this.db.insert(userSessions).values({
      id: generateId(),
      userId,
      tokenHash: tokenHashed,
      deviceInfo: deviceInfo ?? null,
      createdAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
      isRevoked: 0,
    });

    // Sign the token so we can verify without DB lookup for quick checks
    const timestamp = now.getTime().toString();
    const signature = await hmacSign(`${token}:${timestamp}`, this.secret);
    return `${token}:${timestamp}:${signature}`;
  }

  async validateSession(signedToken: string) {
    const parts = signedToken.split(":");
    if (parts.length !== 3) return null;
    const [token, timestamp, signature] = parts;

    // Verify signature
    const valid = await hmacVerify(`${token}:${timestamp}`, signature, this.secret);
    if (!valid) return null;

    // Check expiry
    const ts = parseInt(timestamp);
    if (Date.now() - ts > SESSION_TTL_MS) return null;

    // Lookup in DB
    const tokenHashed = await hashToken(token);
    const [session] = await this.db
      .select()
      .from(userSessions)
      .where(and(eq(userSessions.tokenHash, tokenHashed), eq(userSessions.isRevoked, 0)))
      .limit(1);

    if (!session) return null;
    if (new Date(session.expiresAt) < new Date()) return null;

    const [user] = await this.db
      .select()
      .from(users)
      .where(eq(users.id, session.userId!))
      .limit(1);

    return user ?? null;
  }

  async revokeSession(signedToken: string) {
    const parts = signedToken.split(":");
    if (parts.length < 1) return;
    const tokenHashed = await hashToken(parts[0]);
    await this.db
      .update(userSessions)
      .set({ isRevoked: 1 })
      .where(eq(userSessions.tokenHash, tokenHashed));
  }

  // --- API token management ---

  async createApiToken(userId: string, name: string, scopes?: string[]) {
    const token = `omega_${generateToken()}`;
    const tokenHashed = await hashToken(token);
    await this.db.insert(apiTokens).values({
      id: generateId(),
      userId,
      tokenHash: tokenHashed,
      name,
      scopes: scopes ? JSON.stringify(scopes) : null,
      createdAt: new Date().toISOString(),
      isRevoked: 0,
    });
    return token; // Only returned once
  }

  async validateApiToken(token: string) {
    const tokenHashed = await hashToken(token);
    const [row] = await this.db
      .select()
      .from(apiTokens)
      .where(and(eq(apiTokens.tokenHash, tokenHashed), eq(apiTokens.isRevoked, 0)))
      .limit(1);

    if (!row) return null;
    if (row.expiresAt && new Date(row.expiresAt) < new Date()) return null;

    // Update last_used_at
    await this.db
      .update(apiTokens)
      .set({ lastUsedAt: new Date().toISOString() })
      .where(eq(apiTokens.id, row.id));

    const [user] = await this.db
      .select()
      .from(users)
      .where(eq(users.id, row.userId))
      .limit(1);

    return user ?? null;
  }

  // --- License key validation ---

  async validateLicense(key: string) {
    const [license] = await this.db
      .select()
      .from(licenses)
      .where(and(eq(licenses.key, key), eq(licenses.status, "active")))
      .limit(1);

    if (!license) return null;

    await this.db
      .update(licenses)
      .set({ lastValidatedAt: new Date().toISOString() })
      .where(eq(licenses.id, license.id));

    return license;
  }

  async createProSession(licenseKey: string): Promise<string | null> {
    const license = await this.validateLicense(licenseKey);
    if (!license) return null;

    const timestamp = Date.now().toString();
    const payload = btoa(licenseKey);
    const signature = await hmacSign(`${payload}:${timestamp}`, this.secret);
    return `${payload}:${timestamp}:${signature}`;
  }

  async validateProSession(signedToken: string) {
    const parts = signedToken.split(":");
    if (parts.length !== 3) return null;
    const [payload, timestamp, signature] = parts;

    const valid = await hmacVerify(`${payload}:${timestamp}`, signature, this.secret);
    if (!valid) return null;

    const ts = parseInt(timestamp);
    if (Date.now() - ts > PRO_SESSION_TTL_MS) return null;

    const licenseKey = atob(payload);
    return this.validateLicense(licenseKey);
  }
}

// ===================================================================
// Cookie helpers
// ===================================================================

export function setSessionCookie(token: string): string {
  return `${COOKIE_NAME}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_TTL_MS / 1000}; Secure`;
}

export function setProSessionCookie(token: string): string {
  return `${PRO_COOKIE_NAME}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${PRO_SESSION_TTL_MS / 1000}; Secure`;
}

export function clearSessionCookie(): string {
  return `${COOKIE_NAME}=; Path=/; HttpOnly; Max-Age=0`;
}

export function clearProSessionCookie(): string {
  return `${PRO_COOKIE_NAME}=; Path=/; HttpOnly; Max-Age=0`;
}

export function getSessionFromCookie(cookieHeader: string | null): string | null {
  if (!cookieHeader) return null;
  const match = cookieHeader.match(new RegExp(`${COOKIE_NAME}=([^;]+)`));
  return match?.[1] ?? null;
}

export function getProSessionFromCookie(cookieHeader: string | null): string | null {
  if (!cookieHeader) return null;
  const match = cookieHeader.match(new RegExp(`${PRO_COOKIE_NAME}=([^;]+)`));
  return match?.[1] ?? null;
}

export { COOKIE_NAME, PRO_COOKIE_NAME };

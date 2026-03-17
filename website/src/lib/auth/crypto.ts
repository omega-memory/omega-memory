/**
 * Crypto helpers (Web Crypto API — works on CF Workers + Node 22+)
 */

export async function getSigningKey(secret: string): Promise<CryptoKey> {
  const enc = new TextEncoder();
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

export async function hmacSign(data: string, secret: string): Promise<string> {
  const key = await getSigningKey(secret);
  const enc = new TextEncoder();
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(data));
  return bufToHex(new Uint8Array(sig));
}

export async function hmacVerify(data: string, signature: string, secret: string): Promise<boolean> {
  const expected = await hmacSign(data, secret);
  return timingSafeEqual(expected, signature);
}

export function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

export function bufToHex(buf: Uint8Array): string {
  return Array.from(buf).map(b => b.toString(16).padStart(2, "0")).join("");
}

export async function hashPassword(password: string): Promise<string> {
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

export async function verifyPassword(password: string, stored: string): Promise<boolean> {
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

export function generateId(): string {
  return crypto.randomUUID();
}

export function generateToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return bufToHex(bytes);
}

export async function hashToken(token: string): Promise<string> {
  const enc = new TextEncoder();
  const digest = await crypto.subtle.digest("SHA-256", enc.encode(token));
  return bufToHex(new Uint8Array(digest));
}

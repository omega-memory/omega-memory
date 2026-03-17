/**
 * JWT utilities — Web Crypto API only (CF Workers compatible).
 * No Node.js crypto, no Buffer. All base64url done manually.
 */

// ===================================================================
// Base64url helpers (no Buffer)
// ===================================================================

function base64urlEncode(data: Uint8Array): string {
  const binary = Array.from(data)
    .map((b) => String.fromCharCode(b))
    .join("");
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecode(str: string): Uint8Array {
  // Restore standard base64
  let base64 = str.replace(/-/g, "+").replace(/_/g, "/");
  // Add padding
  while (base64.length % 4 !== 0) {
    base64 += "=";
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function base64urlDecodeToString(str: string): string {
  const bytes = base64urlDecode(str);
  return new TextDecoder().decode(bytes);
}

// ===================================================================
// JWT decode / verify
// ===================================================================

export interface DecodedJwt {
  header: Record<string, unknown>;
  payload: Record<string, unknown>;
  signature: string;
}

/**
 * Decode a JWT without verification (base64url decode only).
 */
export function decodeJwt(token: string): DecodedJwt {
  const parts = token.split(".");
  if (parts.length !== 3) {
    throw new Error("Invalid JWT: expected 3 parts");
  }
  const [headerB64, payloadB64, signatureB64] = parts;
  return {
    header: JSON.parse(base64urlDecodeToString(headerB64)),
    payload: JSON.parse(base64urlDecodeToString(payloadB64)),
    signature: signatureB64,
  };
}

/**
 * Verify HMAC-SHA256 JWT (e.g. for Supabase tokens).
 * Returns the payload if valid, null otherwise.
 */
export async function verifyJwtHmac(
  token: string,
  secret: string
): Promise<Record<string, unknown> | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [headerB64, payloadB64, signatureB64] = parts;
  const signingInput = `${headerB64}.${payloadB64}`;
  const enc = new TextEncoder();

  try {
    const key = await crypto.subtle.importKey(
      "raw",
      enc.encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );

    const sig = await crypto.subtle.sign("HMAC", key, enc.encode(signingInput));
    const expectedSig = base64urlEncode(new Uint8Array(sig));

    // Timing-safe comparison
    if (expectedSig.length !== signatureB64.length) return null;
    let diff = 0;
    for (let i = 0; i < expectedSig.length; i++) {
      diff |= expectedSig.charCodeAt(i) ^ signatureB64.charCodeAt(i);
    }
    if (diff !== 0) return null;

    const payload = JSON.parse(base64urlDecodeToString(payloadB64));

    // Check expiry
    if (payload.exp && typeof payload.exp === "number") {
      if (Date.now() / 1000 > payload.exp) return null;
    }

    return payload;
  } catch {
    return null;
  }
}

/**
 * Verify RSA-signed JWT (RS256) against a JWK public key.
 * Returns the payload if valid, null otherwise.
 */
export async function verifyJwtRsa(
  token: string,
  jwk: JsonWebKey
): Promise<Record<string, unknown> | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [headerB64, payloadB64, signatureB64] = parts;
  const signingInput = `${headerB64}.${payloadB64}`;
  const signatureBytes = base64urlDecode(signatureB64);
  const enc = new TextEncoder();

  try {
    const key = await crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
      false,
      ["verify"]
    );

    const valid = await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      key,
      signatureBytes.buffer as ArrayBuffer,
      enc.encode(signingInput)
    );

    if (!valid) return null;

    const payload = JSON.parse(base64urlDecodeToString(payloadB64));

    // Check expiry
    if (payload.exp && typeof payload.exp === "number") {
      if (Date.now() / 1000 > payload.exp) return null;
    }

    return payload;
  } catch {
    return null;
  }
}

// Simple in-memory cache for JWKS
const jwksCache = new Map<string, { keys: JsonWebKey[]; fetchedAt: number }>();
const JWKS_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

/**
 * Fetch and cache JWKS (JSON Web Key Set) from a URL.
 */
export async function fetchJwks(jwksUri: string): Promise<JsonWebKey[]> {
  const cached = jwksCache.get(jwksUri);
  if (cached && Date.now() - cached.fetchedAt < JWKS_CACHE_TTL_MS) {
    return cached.keys;
  }

  const res = await fetch(jwksUri);
  if (!res.ok) {
    throw new Error(`Failed to fetch JWKS from ${jwksUri}: ${res.status}`);
  }

  const data = (await res.json()) as { keys: JsonWebKey[] };
  const keys = data.keys;
  jwksCache.set(jwksUri, { keys, fetchedAt: Date.now() });
  return keys;
}

/**
 * Find a JWK by kid (key ID) from a JWKS.
 */
export function findJwkByKid(keys: JsonWebKey[], kid: string): JsonWebKey | null {
  return keys.find((k) => (k as Record<string, unknown>).kid === kid) ?? null;
}

export { base64urlEncode, base64urlDecode };

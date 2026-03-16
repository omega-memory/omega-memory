/**
 * Coordination utilities — shared helpers for the coordination tab.
 *
 * SYNC: The AGENT_NAMES list and hashing algorithm must match
 * src/omega/server/hook_server/__init__.py (_AGENT_NAMES) and
 * src/omega/server/hook_server/utils.py (_agent_nickname).
 */

import type { ProjectRecord } from "../hooks/useProjects";

// ── Freshness ────────────────────────────────────────────────────────

export type Freshness = "active" | "idle" | "stale";

/**
 * Stale threshold for hiding sessions by default (15 minutes).
 * Sessions older than this are hidden unless the user toggles "show stale".
 */
export const STALE_THRESHOLD_MS = 15 * 60_000;

/** Determine freshness from a heartbeat timestamp. */
export function getFreshness(heartbeat: string, referenceTime?: number): Freshness {
  const ref = referenceTime ?? Date.now();
  const diff = ref - new Date(heartbeat).getTime();
  const minutes = diff / 60_000;
  if (minutes < 2) return "active";
  if (minutes < 15) return "idle";
  return "stale";
}

// ── Formatting ──────────────────────────────────────────────────────

export function formatUptime(startedAt: string): string {
  const ms = Date.now() - new Date(startedAt).getTime();
  const totalMin = Math.floor(ms / 60_000);
  if (totalMin < 1) return "<1m";
  if (totalMin < 60) return `${totalMin}m`;
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

export function formatTimeAgo(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  const sec = Math.floor(ms / 1000);
  if (sec < 10) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  return `${h}h ago`;
}

// ── Project enrichment ──────────────────────────────────────────────

/** Enrich project name using the project registry, with file path and task text fallbacks. */
export function enrichProject(
  rawProject: string | null | undefined,
  filePaths?: string[],
  taskText?: string | null,
  resolvePathToProject?: (rawPath: string | null | undefined) => ProjectRecord | undefined,
): { name: string; summary?: string } {
  // Try resolver first
  if (resolvePathToProject) {
    const match = resolvePathToProject(rawProject);
    if (match) return { name: match.name };
  }

  // Fallback: find segment after "Projects/" in path
  if (rawProject) {
    const segments = rawProject.replace(/\/+$/, "").split("/");
    const projIdx = segments.indexOf("Projects");
    if (projIdx >= 0 && projIdx + 1 < segments.length) {
      return { name: segments[projIdx + 1], summary: taskText?.slice(0, 60) || undefined };
    }
    // Home directory (e.g. /Users/foo) — show "Home" instead of username
    if (segments.length <= 3 && segments[1] === "Users") {
      return { name: "Home", summary: taskText?.slice(0, 60) || undefined };
    }
    // Non-standard path: use basename
    const last = segments[segments.length - 1];
    if (last && last !== "Unknown") return { name: last };
  }

  // Try to extract project from file paths
  if (filePaths && filePaths.length > 0) {
    for (const fp of filePaths) {
      const segments = fp.split("/");
      const projIdx = segments.findIndex((s) => s === "Projects" || s === "src");
      if (projIdx >= 0 && projIdx + 1 < segments.length) {
        return { name: segments[projIdx + 1], summary: taskText?.slice(0, 60) || undefined };
      }
    }
  }

  // Fall back to task text as summary
  if (taskText) {
    return { name: "Unknown", summary: taskText.slice(0, 60) };
  }

  return { name: "Unknown" };
}

// 90 nature-themed names — same list as Python's _AGENT_NAMES
const AGENT_NAMES = [
  "Alder",   "Aspen",  "Birch",   "Briar",   "Brook",
  "Cedar",   "Cliff",  "Cloud",   "Coral",   "Cove",
  "Crane",   "Creek",  "Dale",    "Dawn",    "Dune",
  "Echo",    "Elm",    "Ember",   "Fern",    "Finch",
  "Flame",   "Flint",  "Flora",   "Fox",     "Frost",
  "Glen",    "Grove",  "Hare",    "Haven",   "Hawk",
  "Hazel",   "Heath",  "Heron",   "Holly",   "Iris",
  "Ivy",     "Jade",   "Jay",     "Juniper", "Lake",
  "Lark",    "Laurel", "Leaf",    "Lily",    "Maple",
  "Marsh",   "Meadow", "Moss",    "Myrtle",  "Oak",
  "Olive",   "Onyx",   "Opal",    "Orca",    "Osprey",
  "Otter",   "Pearl",  "Pebble",  "Pine",    "Plum",
  "Quail",   "Rain",   "Raven",   "Reed",    "Ridge",
  "River",   "Robin",  "Rook",    "Rose",    "Rowan",
  "Rush",    "Sage",   "Shore",   "Sky",     "Slate",
  "Sparrow", "Stone",  "Storm",   "Swift",   "Teal",
  "Thorn",   "Thyme",  "Tide",    "Vale",    "Vine",
  "Violet",  "Willow", "Wren",
] as const;

// Minimal MD5 implementation (RFC 1321).
// Only used for deterministic agent naming — not for security.
function md5(input: string): string {
  const bytes = new TextEncoder().encode(input);

  function toWord(b: Uint8Array, i: number) {
    return (b[i]) | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
  }

  // Pre-processing: pad to 64-byte blocks
  const bitLen = bytes.length * 8;
  const padLen = bytes.length + 1 + ((55 - bytes.length % 64 + 64) % 64);
  const padded = new Uint8Array(padLen + 8);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  // Append original length in bits as 64-bit LE
  padded[padLen]     = bitLen & 0xff;
  padded[padLen + 1] = (bitLen >>> 8) & 0xff;
  padded[padLen + 2] = (bitLen >>> 16) & 0xff;
  padded[padLen + 3] = (bitLen >>> 24) & 0xff;
  // Upper 32 bits of length (for messages < 2^32 bits, these are 0)
  padded[padLen + 4] = 0;
  padded[padLen + 5] = 0;
  padded[padLen + 6] = 0;
  padded[padLen + 7] = 0;

  // Per-round shift amounts
  const s = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ];

  // Pre-computed T table: floor(2^32 * abs(sin(i+1)))
  const K = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
  ];

  let a0 = 0x67452301 >>> 0;
  let b0 = 0xefcdab89 >>> 0;
  let c0 = 0x98badcfe >>> 0;
  let d0 = 0x10325476 >>> 0;

  for (let offset = 0; offset < padded.length; offset += 64) {
    const M: number[] = [];
    for (let j = 0; j < 16; j++) {
      M[j] = toWord(padded, offset + j * 4) >>> 0;
    }

    let A = a0, B = b0, C = c0, D = d0;

    for (let i = 0; i < 64; i++) {
      let F: number, g: number;
      if (i < 16) {
        F = (B & C) | (~B & D);
        g = i;
      } else if (i < 32) {
        F = (D & B) | (~D & C);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        F = B ^ C ^ D;
        g = (3 * i + 5) % 16;
      } else {
        F = C ^ (B | ~D);
        g = (7 * i) % 16;
      }
      F = (F + A + K[i] + M[g]) >>> 0;
      A = D;
      D = C;
      C = B;
      B = (B + ((F << s[i]) | (F >>> (32 - s[i])))) >>> 0;
    }

    a0 = (a0 + A) >>> 0;
    b0 = (b0 + B) >>> 0;
    c0 = (c0 + C) >>> 0;
    d0 = (d0 + D) >>> 0;
  }

  // Output as 32-char hex string (little-endian)
  function toLEHex(v: number) {
    return (
      ((v & 0xff).toString(16).padStart(2, "0")) +
      (((v >>> 8) & 0xff).toString(16).padStart(2, "0")) +
      (((v >>> 16) & 0xff).toString(16).padStart(2, "0")) +
      (((v >>> 24) & 0xff).toString(16).padStart(2, "0"))
    );
  }

  return toLEHex(a0) + toLEHex(b0) + toLEHex(c0) + toLEHex(d0);
}

/**
 * Get the deterministic agent name for a session ID.
 *
 * Algorithm (must match Python _agent_nickname):
 *   MD5(sessionId) -> first 8 hex chars -> parseInt(hex, 16) % NAMES.length
 */
export function getAgentName(sessionId: string): string {
  if (!sessionId) return "Unknown";
  const hash = md5(sessionId);
  const idx = parseInt(hash.slice(0, 8), 16) % AGENT_NAMES.length;
  return AGENT_NAMES[idx];
}

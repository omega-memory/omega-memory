import type { ProjectTimelineEntry, ProjectHealth, ProjectOverview, BoardColumn } from "./types";
import type { CoordinationGitEvent } from "../lib/types";

// --- Noise patterns (matches Python _INTERNAL_TASK_PREFIXES + observed garbage) ---
const NOISE_PATTERNS = [
  /^pre-edit\b/i,
  /^file claim\b/i,
  /^hook invocation\b/i,
  /^read the output file/i,
  /^last login:/i,
  /^welcome to/i,
  /^python \d/i,
  /^\/\w/,               // starts with absolute path
  /^\$ /,                // starts with shell prompt
  /^bash:/i,
  /^error:/i,
  /^warning:/i,
  /^\[.*\]\s*$/,         // just a bracketed status like "[done]"
  /^backfilled:/i,       // auto-generated backfill stubs
  /predated auto-handoff/i,
  // User prompts that leak as session titles
  /^(what'?s?\s*next|whats\s*next)/i,
  /^(says?\s+(i'?m|im)\b)/i,
  /^(help|hi|hello|hey|ok|yes|no|sure|thanks|ty)\s*[.!?]?\s*$/i,
  /^(can you|could you|please|i need|i want)\b/i,
  /^(how do|how to|what is|where is|why is)\b/i,
  /^(check|look at|show me|tell me)\b/i,
  /^(deploy|commit|push|ship|merge)\s*$/i,
  /^(proceed|continue|go ahead|do it|lgtm)\s*$/i,
];

const ACTIVITY_NOISE = [
  ...NOISE_PATTERNS,
  /^auto-claimed\b/i,
  /^session$/i,
  /^claimed\b/i,
  /^merge\b/i,
  /^\d+\s+files?\s+changed/i,
];

const MIN_TASK_LENGTH = 12;

function isNoiseTask(task: string | null | undefined): boolean {
  if (!task) return true;
  if (task.length < MIN_TASK_LENGTH) return true;
  return NOISE_PATTERNS.some((p) => p.test(task));
}

/** Check if an activity summary is noise (too short, generic, or internal). */
export function isNoiseActivity(summary: string): boolean {
  if (!summary || summary.length < 8) return true;
  const cleaned = summary.trim();
  return ACTIVITY_NOISE.some((p) => p.test(cleaned));
}

/**
 * Strip developer jargon: commit prefixes, branch formatting, unicode escapes,
 * task-ID prefixes, and technical shorthand.
 */
export function humanize(raw: string): string {
  let name = raw;

  // Strip everything after literal \n (commit message body/diffstat)
  name = name.split(/\\n/)[0];
  // Also strip after real newlines
  name = name.split(/\n/)[0];

  // Strip diffstat: "5 files changed, 336 insertions(+), 42 deletions(-)"
  name = name.replace(/\d+\s+files?\s+changed.*$/i, "");
  // Strip trailing file counts: ", 94 inser..." or "10 files changed"
  name = name.replace(/,?\s*\d+\s+(insertion|deletion|file).*$/i, "");

  // Strip JSON fragments that leak from tool output
  name = name.replace(/[{["]\s*"?\w+"?\s*:\s*["']?.*$/g, "");
  // Strip trailing JSON keys: ", "stderr": "", "interrupted": false"
  name = name.replace(/,\s*"[^"]+"\s*:.*$/g, "");

  // Decode literal unicode escapes (e.g., \u2605 → ★)
  name = name.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) =>
    String.fromCharCode(parseInt(hex, 16))
  );

  // Strip decorative unicode symbols
  name = name.replace(/[★⭐✦✧◆◇⟠⏭▾▸●○■□▪▫⊕⊗☆♦♢✔✖]/g, "");

  // Strip task-ID prefixes: "PO-LLM:", "P0-DASH:", "P1-SETTINGS:", "#13:", "TASK-42:", etc.
  name = name.replace(/^(P[0-3]|TASK|BUG|FEAT|FIX|ISSUE)[-_]?[A-Z0-9-]*:\s*/i, "");
  // Also strip bare ticket references: "#13-24", "(#13)"
  name = name.replace(/\s*\(#\d+(?:[-–]\d+)?\)/g, "");
  name = name.replace(/\s*#\d+(?:[-–]\d+)?(?:\s|$)/g, " ");

  // Strip conventional commit prefixes: feat(scope):, fix:, etc.
  name = name.replace(
    /^(feat|fix|chore|refactor|docs|test|style|perf|ci|build|revert)(\([^)]*\))?[:\s]+/i,
    ""
  );

  // Strip "Work on branch" / "Working on branch" prefix
  name = name.replace(/^work(?:ing)? on (branch\s+)?/i, "");

  // Strip branch-name prefixes (feat/, fix/, feature/, etc.)
  name = name.replace(/^(feat|fix|chore|feature|hotfix|release|bugfix)\//i, "");

  // Strip commit hashes: "8dc86d3", "Commits: abc1234"
  name = name.replace(/^Commits?:\s*/i, "");
  name = name.replace(/\b[0-9a-f]{7,12}\b/g, "");

  // Strip file paths and extensions (e.g., "Settings.tsx", "route.ts")
  name = name.replace(/\b\w+\.(tsx?|jsx?|py|sql|json|md|css|html)\b/g, (match) => {
    const base = match.replace(/\.\w+$/, "");
    return base.length > 3 ? base : "";
  });

  // Replace common dev shorthand with plain English
  name = name.replace(/\bFK\b/gi, "foreign key");
  name = name.replace(/\benv vars?\b/gi, "settings");
  name = name.replace(/\bUI\b/g, "interface");
  name = name.replace(/\bAPI\b/g, "API");
  name = name.replace(/\bDB\b/g, "database");
  name = name.replace(/\b403\b/g, "access error");
  name = name.replace(/\b404\b/g, "not found");
  name = name.replace(/\b500\b/g, "server error");
  name = name.replace(/\bCRUD\b/gi, "data operations");
  name = name.replace(/\bRPC\b/gi, "server call");

  // Convert underscores to spaces
  name = name.replace(/_/g, " ");

  // Convert CamelCase to spaced words (e.g., "CampaignEditor" -> "Campaign Editor")
  name = name.replace(/([a-z])([A-Z])/g, "$1 $2");

  // If text is all-lowercase-with-hyphens (branch-name style), convert hyphens to spaces
  if (/^[a-z0-9][a-z0-9-]+$/.test(name)) {
    name = name.replace(/-/g, " ");
  }

  // Clean up whitespace
  name = name.replace(/\s+/g, " ").trim();

  // Capitalize first letter
  if (name.length > 0) {
    name = name.charAt(0).toUpperCase() + name.slice(1);
  }

  return name || "Session";
}

/**
 * Simplify a decision string for executive display:
 * take the first sentence, strip jargon, cap length.
 */
export function humanizeDecision(raw: string, maxLen = 100): string {
  // Take first sentence only
  let text = raw.split(/[.!]\s/)[0];
  // Strip strategy/sprint/agent internal details
  text = text.replace(/\b\d+\+?\s*parallel agents?\b/gi, "");
  text = text.replace(/\bstrategy:\s*/gi, "");
  text = text.replace(/\b(file partitioning|contention branch|lock-contention)\b/gi, "");
  text = humanize(text);
  if (text.length > maxLen) text = text.slice(0, maxLen - 3) + "...";
  return text;
}

/**
 * Derive a human-readable display name for a session.
 */
export function cleanTaskName(entry: ProjectTimelineEntry): string {
  const { session, handoff, gitEvents } = entry;

  if (!isNoiseTask(session.task)) return humanize(session.task!);

  if (handoff?.key_context && !isNoiseTask(handoff.key_context)) {
    let ctx = handoff.key_context;
    // Strip memory-engine prefixes that leak from auto-capture
    ctx = ctx.replace(/^(Session handoff:|Auto-checkpoint:|Session summary:|Plan\/decision captured:)\s*/i, "");
    // Strip markdown headers
    ctx = ctx.replace(/^#+\s+/gm, "");
    // Strip leading list markers
    ctx = ctx.replace(/^[-*]\s+/, "");
    // Take first meaningful line
    const firstLine = ctx.split("\n").find((l) => l.trim().length > 10);
    if (firstLine) ctx = firstLine.trim();
    ctx = ctx.length > 100 ? ctx.slice(0, 100) + "…" : ctx;
    return humanize(ctx);
  }

  const completed = parseJsonArray(handoff?.completed_tasks);
  if (completed.length > 0) return humanize(completed[0]);

  // Use first decision as title when available
  const decisionsMade = parseJsonArray(handoff?.decisions_made);
  if (decisionsMade.length > 0) {
    let d = decisionsMade[0];
    // Strip auto-capture prefixes
    d = d.replace(/^(Plan\/decision captured:|Decision:)\s*/i, "").trim();
    if (d.length > 10) return humanize(d.length > 100 ? d.slice(0, 100) + "…" : d);
  }

  const commits = gitEvents.filter((g: CoordinationGitEvent) => g.event_type === "commit");
  if (commits.length > 0 && commits[0].message) {
    return cleanCommitMessage(commits[0].message);
  }

  // Skip generic branch names — "Main" as a title is misleading
  const GENERIC_BRANCHES = ["main", "master", "develop", "release", "staging"];
  if (handoff?.git_branch && !GENERIC_BRANCHES.includes(handoff.git_branch)) {
    return humanize(`Work on ${handoff.git_branch}`);
  }

  return "Session";
}

/**
 * Clean and humanize a git commit message.
 */
export function cleanCommitMessage(raw: string | null | undefined): string {
  if (!raw) return "";
  let msg = raw
    .split(/\\n|[\n\r]/)[0]
    .replace(/\s*Committer:.*$/i, "")
    .replace(/\s*Author:.*$/i, "")
    .replace(/\s*Signed-off-by:.*$/i, "")
    .replace(/\s*Co-Authored-By:.*$/i, "")
    .trim();
  msg = humanize(msg);
  if (msg.length > 120) msg = msg.slice(0, 120) + "…";
  return msg;
}

/**
 * Safely parse a JSON-serialized string array.
 */
export function parseJsonArray(raw: string | null | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Format an ISO timestamp as a relative time string.
 */
export function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return "30d+";
}

/**
 * Get a date string for grouping sessions by day.
 */
export function dateGroup(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const sessionDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diffDays = Math.floor((today.getTime() - sessionDay.getTime()) / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

/**
 * Session status for visual encoding.
 */
export type SessionStatus = "complete" | "active" | "blocked" | "minimal";

export function getSessionStatus(entry: ProjectTimelineEntry): SessionStatus {
  const { handoff, gitEvents } = entry;
  const commits = gitEvents.filter((g: CoordinationGitEvent) => g.event_type === "commit");

  if (!handoff) {
    // No handoff data — infer from git activity
    return commits.length > 0 ? "complete" : "minimal";
  }

  const blocked = parseJsonArray(handoff.blocked_items);
  const completed = parseJsonArray(handoff.completed_tasks);
  const next = parseJsonArray(handoff.next_steps);

  if (blocked.length > 0) return "blocked";
  if (completed.length > 0 && next.length === 0) return "complete";
  if (completed.length > 0 || next.length > 0) return "active";
  // Has a handoff but no task/step data — check for git activity
  if (commits.length > 0) return "complete";
  return "minimal";
}

export const STATUS_THEME = {
  complete: {
    border: "border-l-green-400/40",
    bar: "bg-green-400/70",
    dot: "bg-green-400",
    badge: "bg-green-400/15 text-green-400/70",
    label: "Complete",
  },
  active: {
    border: "border-l-amber-400/30",
    bar: "bg-amber-400/70",
    dot: "bg-amber-400",
    badge: "bg-amber-400/15 text-amber-400/70",
    label: "In Progress",
  },
  blocked: {
    border: "border-l-red-400/30",
    bar: "bg-red-400/60",
    dot: "bg-red-400",
    badge: "bg-red-400/15 text-red-400/70",
    label: "Blocked",
  },
  minimal: {
    border: "border-l-white/[0.06]",
    bar: "bg-white/30",
    dot: "bg-white/25",
    badge: "bg-white/[0.04] text-white/30",
    label: "Session",
  },
} as const;

// ─── Health Theme (project overview) ────────────────────────

export const HEALTH_THEME: Record<
  ProjectHealth,
  { dot: string; label: string; badge: string; border: string }
> = {
  healthy: {
    dot: "bg-green-400",
    label: "Healthy",
    badge: "bg-green-400/15 text-green-400/70",
    border: "border-green-400/20",
  },
  blocked: {
    dot: "bg-red-400",
    label: "Blocked",
    badge: "bg-red-400/15 text-red-400/70",
    border: "border-red-400/20",
  },
  stale: {
    dot: "bg-white/25",
    label: "Stale",
    badge: "bg-white/[0.04] text-white/30",
    border: "border-white/[0.06]",
  },
  attention: {
    dot: "bg-amber-400",
    label: "Needs Attention",
    badge: "bg-amber-400/15 text-amber-400/70",
    border: "border-amber-400/20",
  },
};

/**
 * Build a compact activity summary string: "3 commits, 7 files"
 */
export function getActivitySummary(entry: ProjectTimelineEntry): string {
  const parts: string[] = [];
  const commits = entry.gitEvents.filter((g: CoordinationGitEvent) => g.event_type === "commit");
  const files = parseJsonArray(entry.handoff?.files_modified);
  const tasks = parseJsonArray(entry.handoff?.completed_tasks);
  const decisions = parseJsonArray(entry.handoff?.decisions_made);

  if (tasks.length > 0) parts.push(`${tasks.length} task${tasks.length > 1 ? "s" : ""}`);
  if (decisions.length > 0) parts.push(`${decisions.length} decision${decisions.length > 1 ? "s" : ""}`);
  if (commits.length > 0) parts.push(`${commits.length} commit${commits.length > 1 ? "s" : ""}`);
  if (files.length > 0) parts.push(`${files.length} file${files.length > 1 ? "s" : ""}`);

  // Fallback: if handoff has key_context but no structured data, show a generic label
  if (parts.length === 0 && entry.handoff?.key_context) {
    return "Session activity recorded";
  }

  return parts.join(" \u00b7 ");
}

// ─── Momentum (project overview) ─────────────────────────────

export type Momentum = "up" | "steady" | "cooling" | "stalled";

/**
 * Compare this week's sessions vs avg of prior 3 weeks.
 */
export function computeMomentum(p: ProjectOverview): Momentum {
  if (p.sessionCount30d === 0) return "stalled";
  const recent = p.sessionCount7d;
  const olderAvg = (p.sessionCount30d - p.sessionCount7d) / 3;
  if (olderAvg === 0) return recent > 0 ? "up" : "stalled";
  const ratio = recent / olderAvg;
  if (ratio >= 1.3) return "up";
  if (ratio >= 0.7) return "steady";
  if (ratio >= 0.2) return "cooling";
  return "stalled";
}

export const MOMENTUM_THEME: Record<
  Momentum,
  { icon: string; label: string; cls: string }
> = {
  up:      { icon: "\u2191", label: "momentum", cls: "text-green-400/70" },
  steady:  { icon: "\u2192", label: "steady",   cls: "text-white/30" },
  cooling: { icon: "\u2193", label: "cooling",  cls: "text-amber-400/70" },
  stalled: { icon: "\u26A0", label: "stalled",  cls: "text-red-400/60" },
};

/**
 * Active = has meaningful recent data (sessions, tasks, or commits in last 30d).
 * Projects with zero activity are dormant regardless of health status.
 */
export function isProjectActive(p: ProjectOverview): boolean {
  const hasActivity = p.sessionCount30d > 0 || p.commitCount30d > 0;
  const hasPendingWork = p.tasks.pending + p.tasks.inProgress + p.tasks.blocked > 0;
  return hasActivity || hasPendingWork;
}

/**
 * Context line: prioritize blockers > next steps > summary.
 */
export function getContextLine(
  p: ProjectOverview,
): { text: string; type: "blocker" | "next" | "summary" } | null {
  const handoff = p.latestHandoff;
  if (handoff?.blockedItems?.length) {
    return { text: handoff.blockedItems[0], type: "blocker" };
  }
  if (handoff?.nextSteps?.[0] && handoff.nextSteps[0].length > 15) {
    return { text: handoff.nextSteps[0], type: "next" };
  }
  if (p.summary) return { text: p.summary, type: "summary" };
  return null;
}

// ─── Board Column Assignment ─────────────────────────────────

export function isActiveWithin(p: ProjectOverview, hours: number): boolean {
  if (!p.lastActive) return false;
  return Date.now() - new Date(p.lastActive).getTime() < hours * 3600000;
}

export function getProjectColumn(p: ProjectOverview): BoardColumn {
  if (p.health === "blocked" || p.tasks.blocked > 0) return "blocked";
  if (p.hasActiveSession || isActiveWithin(p, 24)) return "active-now";
  if (isActiveWithin(p, 7 * 24)) return "this-week";
  if (isProjectActive(p)) return "steady";
  return "parked";
}

export function getSparkColor(momentum: Momentum): string {
  return momentum === "up" ? "rgba(74, 222, 128, 0.6)"
    : momentum === "cooling" ? "rgba(251, 191, 36, 0.6)"
    : momentum === "stalled" ? "rgba(248, 113, 113, 0.5)"
    : "rgba(255, 255, 255, 0.3)";
}

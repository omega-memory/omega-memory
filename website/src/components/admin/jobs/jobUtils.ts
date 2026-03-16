// ─── Job Types ──────────────────────────────────────────

export interface Schedule {
  id: string;
  label: string;
  name: string;
  description: string;
  schedule_type: "interval" | "calendar";
  interval_seconds: number | null;
  calendar_hour: number | null;
  calendar_minute: number | null;
  calendar_weekday: number | null;
  enabled: boolean;
  command: string;
  last_status: "ok" | "error" | "unknown";
  last_run_at: string | null;
  pending_changes: Record<string, unknown> | null;
  metadata: ScheduleMetadata | null;
  updated_at: string;
  skip_next: boolean;
}

export interface AccountScheduleDay {
  activity: string;
  reply_quota: string;
}

export interface AccountInfo {
  role: string;
  voice: string;
  schedule: Record<string, AccountScheduleDay>;
  hashtags: string[];
}

export interface ScheduleMetadata {
  accounts?: Record<string, AccountInfo>;
  target_accounts?: {
    tier_1?: string[];
    tier_2?: string[];
    tier_3?: string[];
  };
  rules?: string[];
  queue_file?: string;
  briefing_dir?: string;
  // Cowork scheduled task fields
  runtime?: string;
  cadence?: string;
  prompt_file?: string;
  output_dir?: string;
  omega_goal_id?: string;
  sensors?: string[];
}

// ─── Derived Status ────────────────────────────────────

export type JobStatus = "healthy" | "error" | "late" | "paused" | "new";

/** Grace periods before a job is considered "late" */
const DAILY_GRACE_MS = 30 * 60 * 1000; // 30 min for daily jobs
const WEEKLY_GRACE_MS = 2 * 60 * 60 * 1000; // 2 h for weekly jobs
const INTERVAL_GRACE_MS = 10 * 60 * 1000; // 10 min for interval jobs

export function computeStatus(s: Schedule, recentRuns?: string[]): JobStatus {
  if (!s.enabled) return "paused";

  // Prefer actual run data over the denormalized schedules.last_status cache,
  // which can go stale if sendHeartbeat() fails or a stuck run is reaped.
  const effectiveStatus = recentRuns && recentRuns.length > 0
    ? recentRuns[recentRuns.length - 1] // last element = most recent run
    : s.last_status;

  if (effectiveStatus === "error" && s.last_run_at) return "error";
  if (!s.last_run_at || effectiveStatus === "unknown") return "new";

  const source = (s.metadata as Record<string, unknown>)?.source;

  // Check if overdue
  const nextRun = computeNextRun(s);
  if (nextRun) {
    let grace: number;
    if (source === "maintenance") {
      // Maintenance jobs get 2x their interval as grace (session-triggered, not time-exact)
      grace = (s.interval_seconds ?? 0) * 1000;
    } else if (s.schedule_type === "calendar") {
      grace = s.calendar_weekday != null ? WEEKLY_GRACE_MS : DAILY_GRACE_MS;
    } else {
      grace = INTERVAL_GRACE_MS;
    }
    if (Date.now() > nextRun.getTime() + grace) return "late";
  }

  return "healthy";
}

// ─── Next Run Computation ──────────────────────────────

const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export function computeNextRun(s: Schedule): Date | null {
  if (!s.enabled) return null;

  if (s.schedule_type === "calendar") {
    const h = s.calendar_hour ?? 0;
    const m = s.calendar_minute ?? 0;
    const now = new Date();

    if (s.calendar_weekday != null) {
      // Weekly: find next occurrence of target weekday
      const today = new Date(now);
      today.setHours(h, m, 0, 0);
      const currentDay = now.getDay(); // 0=Sun
      let daysUntil = s.calendar_weekday - currentDay;
      if (daysUntil < 0) daysUntil += 7;
      if (daysUntil === 0 && today.getTime() <= now.getTime()) daysUntil = 7;
      const next = new Date(today);
      next.setDate(next.getDate() + daysUntil);
      return next;
    }

    // Daily: next occurrence of this hour:minute
    const today = new Date(now);
    today.setHours(h, m, 0, 0);
    if (today.getTime() > now.getTime()) return today;
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    return tomorrow;
  }

  if (s.schedule_type === "interval" && s.interval_seconds) {
    if (!s.last_run_at) return null;
    const last = new Date(s.last_run_at);
    if (isNaN(last.getTime())) return null;
    return new Date(last.getTime() + s.interval_seconds * 1000);
  }

  return null;
}

// ─── Formatting ────────────────────────────────────────

export function humanSchedule(s: Schedule): string {
  if (s.schedule_type === "calendar") {
    const h = s.calendar_hour ?? 0;
    const m = s.calendar_minute ?? 0;
    const ampm = h >= 12 ? "PM" : "AM";
    const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
    const time = `${h12}:${m.toString().padStart(2, "0")} ${ampm}`;
    if (s.calendar_weekday != null) {
      return `${DAY_NAMES[s.calendar_weekday]} at ${time}`;
    }
    return `Daily at ${time}`;
  }
  const sec = s.interval_seconds ?? 0;
  if (sec >= 3600) return `Every ${sec / 3600}h`;
  if (sec >= 60) return `Every ${sec / 60} min`;
  return `Every ${sec}s`;
}

export function relativeTime(date: Date | string | null): string {
  if (!date) return "--";
  const d = typeof date === "string" ? new Date(date) : date;
  if (isNaN(d.getTime())) return "--";

  const diffMs = Date.now() - d.getTime();
  const future = diffMs < 0;
  const abs = Math.abs(diffMs);

  const secs = Math.floor(abs / 1000);
  if (secs < 60) return future ? "in <1m" : "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return future ? `in ${mins}m` : `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return future ? `in ${hrs}h` : `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return future ? `in ${days}d` : `${days}d ago`;
}

// ─── Style Helpers ─────────────────────────────────────

export const STATUS_CONFIG: Record<
  JobStatus,
  { label: string; dotCls: string; textCls: string; bgCls: string }
> = {
  healthy: {
    label: "healthy",
    dotCls: "bg-type-lesson",
    textCls: "text-type-lesson",
    bgCls: "bg-type-lesson/[0.12]",
  },
  error: {
    label: "error",
    dotCls: "bg-type-error",
    textCls: "text-type-error",
    bgCls: "bg-type-error/[0.12]",
  },
  late: {
    label: "late",
    dotCls: "bg-type-reminder",
    textCls: "text-type-reminder",
    bgCls: "bg-type-reminder/[0.12]",
  },
  paused: {
    label: "paused",
    dotCls: "bg-ink-tertiary",
    textCls: "text-ink-tertiary",
    bgCls: "bg-ink-tertiary/[0.12]",
  },
  new: {
    label: "new",
    dotCls: "bg-ink-secondary",
    textCls: "text-ink-secondary",
    bgCls: "bg-ink-secondary/[0.12]",
  },
};

export function ownerBadge(label: string): { text: string; cls: string } {
  if (label.startsWith("com.omega.github"))
    return { text: "github", cls: "bg-[#238636]/15 text-[#58a65c]" };
  if (label.startsWith("com.omega.vercel"))
    return { text: "vercel", cls: "bg-[#0070f3]/15 text-[#4da3ff]" };
  if (label.startsWith("com.omega.maintenance"))
    return { text: "maint", cls: "bg-type-observation/15 text-type-observation" };
  if (label.startsWith("com.omega.cowork"))
    return { text: "cowork", cls: "bg-type-lesson/15 text-type-lesson" };
  if (label.startsWith("com.omega"))
    return { text: "omega", cls: "bg-gold/[0.12] text-gold" };
  if (label.startsWith("com.claude"))
    return { text: "claude", cls: "bg-type-decision/15 text-type-decision" };
  if (label.startsWith("com.magma"))
    return {
      text: "magma",
      cls: "bg-type-preference/15 text-type-preference",
    };
  return { text: "system", cls: "bg-surface-elevated text-ink-tertiary" };
}

/** Sources where the schedule is defined in code and not editable from UI. */
const READ_ONLY_SOURCES = new Set(["github_actions", "vercel_cron", "maintenance"]);

export function isReadOnly(s: Schedule): boolean {
  const source = (s.metadata as Record<string, unknown>)?.source;
  return typeof source === "string" && READ_ONLY_SOURCES.has(source);
}

export function sourceLabel(s: Schedule): string | null {
  const source = (s.metadata as Record<string, unknown>)?.source;
  if (source === "github_actions") return "Defined in GitHub Actions workflow";
  if (source === "vercel_cron") return "Defined in vercel.json";
  if (source === "maintenance") return "Session-triggered maintenance stage";
  return null;
}

// ─── Schedule Runs ──────────────────────────────────────

export interface ScheduleRun {
  id: string;
  schedule_label: string;
  status: "ok" | "error" | "running" | "dead_letter" | "pending_approval" | "approved" | "rejected";
  started_at: string;
  finished_at: string | null;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: string;
  error: string | null;
  attempt?: number;
  max_attempts?: number;
  next_retry_at?: string | null;
  dead_letter_at?: string | null;
  acknowledged_at?: string | null;
  acknowledged_by?: string | null;
  idempotency_key?: string | null;
}

export { formatCost, formatTokens } from "../lib/format";

export function formatDuration(ms: number): string {
  if (ms <= 0) return "--";
  const secs = Math.round(ms / 1000);
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  const remSecs = secs % 60;
  if (mins < 60) return remSecs > 0 ? `${mins}m${remSecs}s` : `${mins}m`;
  const hrs = Math.floor(mins / 60);
  const remMins = mins % 60;
  return remMins > 0 ? `${hrs}h${remMins}m` : `${hrs}h`;
}

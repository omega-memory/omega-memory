import { parseContentRich, isConfidentialContent } from "../contentUtils";
import type { ParsedContent } from "../contentUtils";

// ─── Types ──────────────────────────────────────────────────

export interface Memory {
  id: string;
  content: string;
  event_type: string | null;
  priority: number;
  created_at: string;
  project: string | null;
  metadata: Record<string, any> | null;
}

// ─── Date & Urgency Helpers ─────────────────────────────────

export function formatDue(date: Date): string {
  const diff = Math.floor(
    (date.getTime() - Date.now()) / 86400000
  );
  if (diff < -1) return `${Math.abs(diff)}d overdue`;
  if (diff === -1) return "Yesterday";
  if (diff === 0) return "Due today";
  if (diff === 1) return "Tomorrow";
  if (diff <= 7) return `In ${diff} days`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function groupByDateSegment(memories: Memory[]): [string, Memory[]][] {
  const groups = new Map<string, Memory[]>();
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const dayOfWeek = today.getDay();
  const mondayOffset = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const thisWeekStart = new Date(today);
  thisWeekStart.setDate(thisWeekStart.getDate() - mondayOffset);

  const lastWeekStart = new Date(thisWeekStart);
  lastWeekStart.setDate(lastWeekStart.getDate() - 7);

  for (const m of memories) {
    const d = new Date(m.created_at);
    const dDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    let label: string;

    // Guard against invalid dates (epoch 0, NaN, pre-2020)
    if (isNaN(d.getTime()) || d.getFullYear() < 2020) {
      label = "Today";
    } else if (dDay.getTime() === today.getTime()) {
      label = "Today";
    } else if (dDay.getTime() === yesterday.getTime()) {
      label = "Yesterday";
    } else if (dDay >= thisWeekStart) {
      label = "This Week";
    } else if (dDay >= lastWeekStart) {
      label = "Last Week";
    } else {
      label = d.toLocaleDateString("en-US", { month: "long", year: "numeric" });
    }

    const list = groups.get(label) || [];
    list.push(m);
    groups.set(label, list);
  }
  return Array.from(groups.entries());
}

// ─── Importance Scoring ─────────────────────────────────────

export function computeImportanceScore(m: Memory): number {
  let score = (m.priority || 1) * 20;

  const ageMs = Date.now() - new Date(m.created_at).getTime();
  const ageH = ageMs / 3600000;
  if (ageH < 24) score += 15;
  else if (ageH < 72) score += 5;
  else if (ageH > 168) score -= 10;
  else if (ageH > 48) score -= 5;

  const typeBoosts: Record<string, number> = {
    reminder: 12,
    decision: 10,
    error_pattern: 8,
    lesson_learned: 7,
    session_summary: 6,
    task_completion: 5,
    user_preference: 4,
  };
  if (m.event_type && typeBoosts[m.event_type]) score += typeBoosts[m.event_type];

  if (m.metadata?.tag) score += 8;

  const parsed = parseContentRich(m.content);
  if (parsed.dueDate) {
    const daysUntil = (parsed.dueDate.getTime() - Date.now()) / 86400000;
    if (daysUntil < 0) score += 25;
    else if (daysUntil <= 3) score += 20;
    else if (daysUntil <= 7) score += 10;
  }

  if (m.metadata?.remind_at) {
    const remindAt = new Date(m.metadata.remind_at).getTime();
    const daysUntil = (remindAt - Date.now()) / 86400000;
    if (daysUntil < 0) score += 25;
    else if (daysUntil <= 1) score += 15;
  }

  return score;
}

/** Explicit actionability predicate: replaces score-based threshold */
export function needsAttention(m: Memory): boolean {
  const meta = m.metadata || {};

  // Blocklist: never qualifies
  if (["session_summary", "task_completion", "user_preference"].includes(m.event_type || "")) return false;
  if (meta.reminder_status === "dismissed") return false;
  if (meta.archived_at) return false;

  // Session recap with unread next steps
  if (m.event_type === "session_recap" && meta.session_achievement?.has_unread_next_steps) return true;

  // Active reminder overdue or due within 24h
  if (m.event_type === "reminder" && meta.remind_at) {
    const hoursUntil = (new Date(meta.remind_at).getTime() - Date.now()) / 3_600_000;
    if (hoursUntil <= 24) return true;
  }

  // Error pattern < 72h old without relevance status
  if (m.event_type === "error_pattern" && !meta.relevance_status) {
    const ageH = (Date.now() - new Date(m.created_at).getTime()) / 3_600_000;
    if (ageH < 72) return true;
  }

  // Overdue due date
  const parsed = parseContentRich(m.content);
  if (parsed.dueDate) {
    const daysUntil = (parsed.dueDate.getTime() - Date.now()) / 86_400_000;
    if (daysUntil < 0) return true;
    if (daysUntil <= 3 && !meta.read_at) return true;
  }

  return false;
}

// ─── Expansion Mode ──────────────────────────────────────────

export type ExpansionMode = "full" | "inline" | "none";

/** Determine how an item should expand when clicked */
export function getExpansionMode(m: Memory, parsed: ParsedContent): ExpansionMode {
  const meta = m.metadata || {};

  // "full": items with actionable or rich expanded content
  if (meta.is_session_recap) return "full";                                // session recap cards
  if (m.event_type === "reminder") return "full";                          // reminders need snooze/dismiss
  if (isConfidentialContent(m.content, m.metadata)) return "full";         // needs unlock UI
  if (meta.ratable && !meta.rating) return "full";                         // needs rating UI

  // Relevance check eligible
  const ageH = (Date.now() - new Date(m.created_at).getTime()) / 3_600_000;
  if (
    ["decision", "lesson_learned", "error_pattern"].includes(m.event_type || "") &&
    ageH >= 12 &&
    !meta.relevance_status
  ) return "full";

  // Research/benchmark reports
  if (["research_report", "sota_research", "benchmark_update"].includes(m.event_type || "")) return "full";

  // Structurally rich content
  if (parsed.sections.length > 0) return "full";
  if (parsed.metrics.length >= 2 || parsed.keyValues.length >= 2) return "full";
  if (parsed.bullets.length > 5) return "full";

  // Source URL items (ingested content)
  if (meta.source_url) return "full";

  // "inline": has meaningful detail text worth showing
  if (parsed.detail && parsed.detail.length > 20) return "inline";
  if (m.content.length > 200) return "inline";

  // "none": short decisions, preferences, task completions
  return "none";
}

export function splitByImportance(memories: Memory[]): {
  important: Memory[];
  rest: Memory[];
} {
  const important = memories
    .filter(needsAttention)
    .sort((a, b) => computeImportanceScore(b) - computeImportanceScore(a))
    .slice(0, 8);

  const importantIds = new Set(important.map((m) => m.id));
  const rest = memories.filter((m) => !importantIds.has(m.id));

  return { important, rest };
}

export function groupByDaySorted(memories: Memory[]): [string, Memory[]][] {
  const groups = groupByDateSegment(memories);
  return groups.map(([label, items]) => {
    const sorted = [...items].sort(
      (a, b) => computeImportanceScore(b) - computeImportanceScore(a)
    );
    return [label, sorted];
  });
}

export function sortByScore(memories: Memory[]): Memory[] {
  return [...memories].sort(
    (a, b) => computeImportanceScore(b) - computeImportanceScore(a)
  );
}

// ─── Filters ────────────────────────────────────────────────

// ─── Outcome-oriented Filters ────────────────────────────────

export const PRIMARY_FILTERS: { value: string; label: string }[] = [
  { value: "all", label: "All" },
  { value: "action_required", label: "Action Required" },
  { value: "decision", label: "Decisions" },
  { value: "error_pattern", label: "Issues" },
  { value: "insights", label: "Insights" },
  { value: "progress", label: "Progress" },
];

export const SECONDARY_FILTERS: { value: string; label: string }[] = [
  { value: "session_recap", label: "Sessions" },
  { value: "reminder", label: "Reminders" },
  { value: "x_brief", label: "X Brief" },
  { value: "url_ingest", label: "Ingested" },
  { value: "tagged", label: "Flagged" },
  { value: "deadlines", label: "Due Dates" },
  { value: "grants", label: "Grants" },
];

// Backward compat aliases
export const SERVER_FILTERS = PRIMARY_FILTERS;
export const CLIENT_FILTERS = SECONDARY_FILTERS;

export type SortOption = "priority" | "newest" | "oldest" | "unread";

// ─── Outcome Filter Predicates ───────────────────────────────

/** Items requiring operator action: extends needsAttention + relevant lessons */
export function isActionRequired(m: Memory): boolean {
  if (needsAttention(m)) return true;
  // Lesson learned that hasn't been reviewed
  if (m.event_type === "lesson_learned" && !m.metadata?.relevance_status) {
    const ageH = (Date.now() - new Date(m.created_at).getTime()) / 3_600_000;
    if (ageH < 168) return true; // within a week
  }
  return false;
}

/** Insights: lessons, research, benchmarks */
export function isInsight(m: Memory): boolean {
  return ["lesson_learned", "research_report", "sota_research", "benchmark_update"].includes(m.event_type || "");
}

/** Progress: tasks, session summaries, session recaps */
export function isProgress(m: Memory): boolean {
  return ["task_completion", "session_summary", "session_recap"].includes(m.event_type || "");
}

/** Map filter value to server-side event_type (null = client-side filter) */
export function filterToEventType(filter: string): string | null {
  const serverTypes: Record<string, string> = {
    decision: "decision",
    error_pattern: "error_pattern",
    reminder: "reminder",
  };
  return serverTypes[filter] || null;
}

// ─── Type-Specific Visual System ─────────────────────────────

export interface TypeVisual {
  iconPath: string;
  colorClass: string;
  bgTint: string;
  label: string;
}

export const TYPE_VISUALS: Record<string, TypeVisual> = {
  decision: {
    iconPath: "M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z",
    colorClass: "text-type-decision",
    bgTint: "card-bg-decision",
    label: "Decision",
  },
  lesson_learned: {
    iconPath: "M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456Z",
    colorClass: "text-type-lesson",
    bgTint: "card-bg-lesson",
    label: "Learned",
  },
  error_pattern: {
    iconPath: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z",
    colorClass: "text-type-error",
    bgTint: "card-bg-error",
    label: "Issue",
  },
  reminder: {
    iconPath: "M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0",
    colorClass: "text-type-reminder",
    bgTint: "card-bg-reminder",
    label: "Reminder",
  },
  task_completion: {
    iconPath: "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
    colorClass: "text-type-task",
    bgTint: "card-bg-task",
    label: "Done",
  },
  user_preference: {
    iconPath: "M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75",
    colorClass: "text-type-preference",
    bgTint: "card-bg-preference",
    label: "Setting",
  },
  session_summary: {
    iconPath: "M6.429 9.75L2.25 12l4.179 2.25m0-4.5l5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L21.75 12l-4.179 2.25m0 0L12 16.5l-5.571-2.25m11.142 0L21.75 16.5 12 21.75 2.25 16.5l4.179-2.25",
    colorClass: "text-type-session",
    bgTint: "card-bg-session",
    label: "Recap",
  },
  session_recap: {
    iconPath: "M15.59 14.37a6 6 0 0 1-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 0 0 6.16-12.12A14.98 14.98 0 0 0 9.631 8.41m5.96 5.96a14.926 14.926 0 0 1-5.841 2.58m-.119-8.54a6 6 0 0 0-7.381 5.84h4.8m2.58-5.84a14.927 14.927 0 0 1-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 0 1-2.448-2.448 14.9 14.9 0 0 1 .06-.312m-2.24 2.39a4.493 4.493 0 0 0-1.757 4.306 4.493 4.493 0 0 0 4.306-1.758M16.5 9a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0Z",
    colorClass: "text-type-session",
    bgTint: "card-bg-session",
    label: "Session",
  },
};

export const XBRIEF_VISUAL: TypeVisual = {
  iconPath: "M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 0 1-2.25 2.25M16.5 7.5V18a2.25 2.25 0 0 0 2.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 0 0 2.25 2.25h13.5M6 7.5h3v3H6v-3Z",
  colorClass: "text-sky-400",
  bgTint: "card-bg-xbrief",
  label: "X Brief",
};

export const GRANT_VISUAL: TypeVisual = {
  iconPath: "M12 6v12m-3-2.818.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  colorClass: "text-type-reminder",
  bgTint: "card-bg-grant",
  label: "Grant",
};

export const DEFAULT_VISUAL: TypeVisual = {
  iconPath: "M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z",
  colorClass: "text-ink-faint",
  bgTint: "",
  label: "Note",
};

export function getTypeVisual(
  eventType: string | null,
  cardType: string | undefined,
  isXBrief: boolean,
  isGrant: boolean,
): TypeVisual {
  if (isXBrief) return XBRIEF_VISUAL;
  if (isGrant) return GRANT_VISUAL;
  return TYPE_VISUALS[eventType || ""] || DEFAULT_VISUAL;
}

// ─── Shared SVG Icon ────────────────────────────────────────

export function TypeIcon({ d, className }: { d: string; className?: string }) {
  return (
    <svg className={className || "w-4 h-4"} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

// ─── Supabase Mutations ─────────────────────────────────────

export async function dismissReminder(
  id: string,
  metadata: Record<string, any> | null
): Promise<boolean> {
  const updated = {
    ...(metadata || {}),
    reminder_status: "dismissed",
    dismissed_from: "web",
    dismissed_at: new Date().toISOString(),
  };
  try {
    const res = await fetch("/api/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, metadata: updated }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function extendReminder(
  id: string,
  metadata: Record<string, any> | null,
  hours = 1
): Promise<{ success: boolean; newRemindAt: string | null }> {
  const currentRemindAt = metadata?.remind_at
    ? new Date(metadata.remind_at).getTime()
    : Date.now();
  const base = Math.max(currentRemindAt, Date.now());
  const newRemindAt = new Date(base + hours * 3600000).toISOString();

  const updated = {
    ...(metadata || {}),
    remind_at: newRemindAt,
    extended_from: "web",
    extended_at: new Date().toISOString(),
  };
  try {
    const res = await fetch("/api/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, metadata: updated }),
    });
    return { success: res.ok, newRemindAt: res.ok ? newRemindAt : null };
  } catch {
    return { success: false, newRemindAt: null };
  }
}

// ─── Read / Archive Mutations ────────────────────────────────

export async function markAsRead(
  id: string,
  metadata: Record<string, any> | null
): Promise<boolean> {
  const updated = { ...(metadata || {}), read_at: new Date().toISOString() };
  try {
    const res = await fetch("/api/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, metadata: updated }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function archiveMemory(
  id: string,
  metadata: Record<string, any> | null
): Promise<boolean> {
  const updated = {
    ...(metadata || {}),
    archived_at: new Date().toISOString(),
    archived_from: "web",
  };
  try {
    const res = await fetch("/api/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, metadata: updated }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function unarchiveMemory(
  id: string,
  metadata: Record<string, any> | null
): Promise<boolean> {
  const updated = { ...(metadata || {}) };
  delete updated.archived_at;
  delete updated.archived_from;
  try {
    const res = await fetch("/api/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, metadata: updated }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ─── Relevance Check Config ─────────────────────────────────

export const RELEVANCE_CHECK_TYPES = ["decision", "lesson_learned", "error_pattern"];

export const RELEVANCE_OPTIONS: Record<string, { label: string; options: string[] }> = {
  decision: { label: "Is this decision still current?", options: ["Still active", "Needs update", "Resolved"] },
  lesson_learned: { label: "Has this lesson been applied?", options: ["Applied", "Worth keeping", "No longer relevant"] },
  error_pattern: { label: "Is this still happening?", options: ["Still occurring", "Workaround found", "Fixed"] },
};

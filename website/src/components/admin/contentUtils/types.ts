// ─── Types ──────────────────────────────────────────────────

export type ContentCategory =
  | "checkpoint"
  | "benchmark"
  | "entity"
  | "grant"
  | "financial"
  | "deadline"
  | "stats"
  | "general";

export interface ParsedContent {
  title: string;
  detail: string | null;
  dueDate: Date | null;
  category: ContentCategory;
  keyValues: { key: string; value: string }[];
  metrics: { label: string; value: string; unit?: string }[];
  sections: { heading: string; body: string }[];
  bullets: string[];
}

// ─── Constants ──────────────────────────────────────────────

export const TYPE_META: Record<string, { label: string; bar: string }> = {
  decision: { label: "Decision", bar: "bg-type-decision" },
  lesson_learned: { label: "Learned", bar: "bg-type-lesson" },
  user_preference: { label: "Setting", bar: "bg-type-preference" },
  error_pattern: { label: "Issue", bar: "bg-type-error" },
  session_summary: { label: "Recap", bar: "bg-type-session" },
  task_completion: { label: "Done", bar: "bg-type-task" },
  reminder: { label: "Reminder", bar: "bg-type-reminder" },
  poll: { label: "Poll", bar: "bg-type-preference" },
  context_prompt: { label: "Question", bar: "bg-type-decision" },
  preference: { label: "Preference", bar: "bg-type-preference" },
  benchmark_update: { label: "Results", bar: "bg-type-lesson" },
  research_report: { label: "Research", bar: "bg-type-decision" },
  sota_research: { label: "Research", bar: "bg-type-decision" },
};

export const DEFAULT_META = { label: "Note", bar: "bg-ink-faint" };

export const TYPE_VERBS: Record<string, string> = {
  decision: "Decided",
  lesson_learned: "Learned",
  benchmark_update: "Results",
  research_report: "Researched",
  sota_research: "Researched",
  task_completion: "Finished",
  error_pattern: "Found issue",
  reminder: "Reminder",
  user_preference: "Updated",
  session_summary: "Recap",
};

export const SECTION_LABELS: Record<string, string> = {
  "files changed": "What Changed",
  "files_changed": "What Changed",
  "next steps": "Up Next",
  "next_steps": "Up Next",
  "plan": "Plan",
  "progress": "Progress",
  "decisions": "Decisions Made",
  "key context": "Background",
  "key_context": "Background",
  "blockers": "Blocked By",
  "dependencies": "Depends On",
  "summary": "Overview",
  "details": "Details",
  "implementation": "How It Works",
};

export const PROJECT_NAMES: Record<string, string> = {
  omega: "OmegaMax",
  element1: "Element1",
  kokyo: "Kokyo",
  "claude-mcp-swift": "Gnosis",
  "polymarket-omega": "Polymarket",
  "jason-sosa-website": "jasonsosa.com",
  "personal-assistant": "Assistant",
  memorystress: "MemoryStress",
  "lightning-memory": "Lightning",
  "marketing-studio-dashboard": "Studio",
  "property-management": "PropMgmt",
  "email-marketing": "Email Mktg",
};

// ─── Grant Types ────────────────────────────────────────────

export interface GrantData {
  status: "submitted" | "draft" | "pending" | "approved" | "rejected" | null;
  amount: string | null;
  score: { value: number; max: number } | null;
  funder: string | null;
  duration: string | null;
  deadline: string | null;
}

export const GRANT_STATUS_STYLES: Record<string, { dot: string; bg: string; text: string; label: string }> = {
  submitted: { dot: "bg-type-lesson", bg: "bg-type-lesson/[0.08]", text: "text-type-lesson", label: "Submitted" },
  approved: { dot: "bg-emerald-400", bg: "bg-emerald-400/[0.08]", text: "text-emerald-400", label: "Approved" },
  rejected: { dot: "bg-type-error", bg: "bg-type-error/[0.08]", text: "text-type-error", label: "Rejected" },
  pending: { dot: "bg-type-reminder", bg: "bg-type-reminder/[0.08]", text: "text-type-reminder", label: "Pending" },
  draft: { dot: "bg-ink-tertiary", bg: "bg-ink-tertiary/[0.08]", text: "text-ink-tertiary", label: "Draft" },
};

// ─── Leaderboard Types ──────────────────────────────────────

export interface LeaderboardEntry {
  rank: number;
  name: string;
  score: number | null;
  isOmega: boolean;
}

// ─── Consolidated Type Registry ─────────────────────────────

export const TYPE_REGISTRY: Record<string, {
  label: string;
  color: string;
  colorClass: string;
  bgTint: string;
  barClass: string;
  iconPath: string;
  description: string;
}> = {
  decision: {
    label: "Decisions",
    color: "#6b9fff",
    colorClass: "text-type-decision",
    bgTint: "card-bg-decision",
    barClass: "bg-type-decision",
    iconPath: "M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z",
    description: "Key outcomes, architectural choices, and deployment decisions captured during sessions",
  },
  lesson_learned: {
    label: "Lessons",
    color: "#5ec9a0",
    colorClass: "text-type-lesson",
    bgTint: "card-bg-lesson",
    barClass: "bg-type-lesson",
    iconPath: "M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456Z",
    description: "Debugging insights, patterns, and reusable knowledge extracted from experience",
  },
  error_pattern: {
    label: "Issues",
    color: "#f06060",
    colorClass: "text-type-error",
    bgTint: "card-bg-error",
    barClass: "bg-type-error",
    iconPath: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z",
    description: "Recurring errors and their fixes, used to short-circuit future debugging",
  },
  reminder: {
    label: "Reminders",
    color: "#e8a040",
    colorClass: "text-type-reminder",
    bgTint: "card-bg-reminder",
    barClass: "bg-type-reminder",
    iconPath: "M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0",
    description: "Time-based reminders and follow-ups scheduled across sessions",
  },
  task_completion: {
    label: "Tasks",
    color: "#40c8c8",
    colorClass: "text-type-task",
    bgTint: "card-bg-task",
    barClass: "bg-type-task",
    iconPath: "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
    description: "Completed tasks and their outcomes, used for progress tracking",
  },
  user_preference: {
    label: "Preferences",
    color: "#b088e8",
    colorClass: "text-type-preference",
    bgTint: "card-bg-preference",
    barClass: "bg-type-preference",
    iconPath: "M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75",
    description: "Your stated preferences: tools, workflows, communication style",
  },
  session_summary: {
    label: "Sessions",
    color: "#7878a0",
    colorClass: "text-type-session",
    bgTint: "card-bg-session",
    barClass: "bg-type-session",
    iconPath: "M6.429 9.75L2.25 12l4.179 2.25m0-4.5l5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L21.75 12l-4.179 2.25m0 0L12 16.5l-5.571-2.25m11.142 0L21.75 16.5 12 21.75 2.25 16.5l4.179-2.25",
    description: "End-of-session summaries capturing what was accomplished",
  },
  unknown: {
    label: "Other",
    color: "#555",
    colorClass: "text-ink-faint",
    bgTint: "",
    barClass: "bg-ink-faint",
    iconPath: "M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z",
    description: "Uncategorized memories",
  },
};

export const SUBSYSTEM_COLORS: Record<string, string> = {
  retrieval: "bg-blue-500/15 text-blue-400 border-blue-500/25",
  storage: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  embedding: "bg-violet-500/15 text-violet-400 border-violet-500/25",
  coordination: "bg-amber-500/15 text-amber-400 border-amber-500/25",
  protocol: "bg-rose-500/15 text-rose-400 border-rose-500/25",
  entity: "bg-cyan-500/15 text-cyan-400 border-cyan-500/25",
  knowledge: "bg-orange-500/15 text-orange-400 border-orange-500/25",
  mcp: "bg-indigo-500/15 text-indigo-400 border-indigo-500/25",
  hooks: "bg-pink-500/15 text-pink-400 border-pink-500/25",
};

export const SOURCE_ICONS: Record<string, string> = {
  arxiv: "Paper",
  blog: "Blog",
  tweet: "Tweet",
  github: "GitHub",
  conference: "Conf",
  news: "News",
  url_ingest: "URL",
  unknown: "Other",
};

export const PLAN_STATUS_STYLES: Record<string, string> = {
  draft: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
  ready: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  implemented: "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

export const FILTERS = [
  "all",
  "starred",
  "high-priority",
  "scanner",
  "manual",
  "plans",
  "dismissed",
] as const;
export type Filter = (typeof FILTERS)[number];

export const FILTER_LABELS: Record<Filter, string> = {
  all: "All",
  starred: "Starred",
  "high-priority": "High Pri",
  scanner: "Scanner",
  manual: "Manual",
  plans: "Plans",
  dismissed: "Dismissed",
};

export function timeAgo(dateStr: string | null | undefined): string {
  if (!dateStr) return "";
  const ts = new Date(dateStr).getTime();
  if (isNaN(ts)) return "";
  const diff = Date.now() - ts;
  if (diff < 0) return "just now";
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

export function priorityColor(score: number): string {
  if (score >= 20) return "text-type-error font-bold";
  if (score >= 12) return "text-type-reminder font-semibold";
  if (score >= 6) return "text-ink-secondary";
  return "text-ink-faint";
}

export function priorityBg(score: number): string {
  if (score >= 20) return "bg-type-error/10 border-type-error/20";
  if (score >= 12) return "bg-type-reminder/10 border-type-reminder/20";
  return "bg-surface-elevated border-edge";
}

/** Priority as a percentage (0-100) for visual bars */
export function priorityPercent(score: number): number {
  return Math.min(100, Math.round((score / 25) * 100));
}

/** Color for the priority bar fill */
export function priorityBarColor(score: number): string {
  if (score >= 20) return "bg-red-500";
  if (score >= 15) return "bg-amber-500";
  if (score >= 12) return "bg-yellow-500";
  if (score >= 8) return "bg-blue-500";
  return "bg-zinc-500";
}

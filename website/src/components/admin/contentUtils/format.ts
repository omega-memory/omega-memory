import { SECTION_LABELS, PROJECT_NAMES } from "./types";

// ─── Section Heading Humanization ───────────────────────────

export function humanizeSectionHeading(heading: string): string {
  return SECTION_LABELS[heading.toLowerCase().trim()] || heading.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Project Name Humanization ──────────────────────────────

export function humanProject(project: string | null): string | null {
  if (!project) return null;
  const slug = project.split("/").filter(Boolean).pop() || "";
  return PROJECT_NAMES[slug.toLowerCase()] || slug;
}

// ─── Time Formatting ────────────────────────────────────────

export function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "Just now";
  const d = new Date(dateStr);
  if (isNaN(d.getTime()) || d.getFullYear() < 2020) return "Just now";
  const secs = Math.floor(
    (Date.now() - d.getTime()) / 1000
  );
  if (secs < 60) return "Just now";
  if (secs < 3600) {
    const m = Math.floor(secs / 60);
    return `${m} min ago`;
  }
  if (secs < 86400) {
    const h = Math.floor(secs / 3600);
    return h === 1 ? "1 hour ago" : `${h} hours ago`;
  }
  if (secs < 172800) return "Yesterday";
  if (secs < 604800) {
    const d = Math.floor(secs / 86400);
    return `${d} days ago`;
  }
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// ─── Shared Utilities (used by Dashboard, Feed, Insights) ───

export function timeUntil(iso: string | null): string {
  if (!iso) return "";
  const ms = new Date(iso).getTime() - Date.now();
  if (ms <= 0) return "now";
  const min = Math.floor(ms / 60000);
  if (min < 60) return `in ${min}m`;
  const hrs = Math.floor(min / 60);
  if (hrs < 24) return `in ${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "tomorrow";
  return `in ${days}d`;
}

export function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export function statusDot(status: string): string {
  if (status === "ok") return "bg-type-lesson shadow-[0_0_6px_rgba(94,201,160,0.4)]";
  if (status === "error") return "bg-type-error shadow-[0_0_6px_rgba(240,96,96,0.4)]";
  if (status === "overdue") return "bg-type-reminder shadow-[0_0_6px_rgba(245,158,11,0.4)]";
  return "bg-ink-faint";
}

export function humanSchedule(s: { scheduleType: string; calendarHour?: number; calendarMinute?: number; intervalSeconds?: number }): string {
  if (s.scheduleType === "calendar") {
    const h = s.calendarHour ?? 0;
    const m = s.calendarMinute ?? 0;
    const ampm = h >= 12 ? "PM" : "AM";
    const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
    return `Daily at ${h12}:${m.toString().padStart(2, "0")} ${ampm}`;
  }
  const sec = s.intervalSeconds ?? 0;
  if (sec >= 3600) return `Every ${sec / 3600}h`;
  if (sec >= 60) return `Every ${sec / 60} min`;
  return `Every ${sec}s`;
}

export function ictDate(): string {
  return new Date().toLocaleDateString("en-US", {
    timeZone: "Asia/Bangkok",
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function ictTime(): string {
  return new Date().toLocaleTimeString("en-US", {
    timeZone: "Asia/Bangkok",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRelativeTime(date: Date): string {
  const diffMs = date.getTime() - Date.now();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays < -1) return `overdue by ${Math.abs(diffDays)} days`;
  if (diffDays === -1) return 'due yesterday';
  if (diffDays === 0) return 'due today';
  if (diffDays === 1) return 'due tomorrow';
  if (diffDays <= 7) return `in ${diffDays} days`;
  return `due ${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
}

import { useCallback, useEffect, useState } from "react";
import { timeAgo } from "./contentUtils";

// ─── Types ──────────────────────────────────────────────────

interface ReminderItem {
  id: string;
  content: string;
  priority: number;
  created_at: string;
  project: string | null;
  metadata: {
    due_date?: string;
    status?: string;
    snoozed_until?: string;
    completed_at?: string;
    [key: string]: unknown;
  } | null;
}

// ─── Helpers ────────────────────────────────────────────────

function formatDueDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = d.getTime() - now.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays < -1) return `${Math.abs(diffDays)}d overdue`;
  if (diffDays === -1) return "Yesterday";
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Tomorrow";
  if (diffDays <= 7) return `In ${diffDays}d`;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function isOverdue(item: ReminderItem): boolean {
  const due = item.metadata?.due_date;
  if (!due) return false;
  return new Date(due).getTime() < Date.now();
}

function priorityBadge(priority: number): { text: string; color: string; bg: string } {
  if (priority >= 8) return { text: "URGENT", color: "text-type-error", bg: "bg-type-error/10" };
  if (priority >= 5) return { text: "IMPORTANT", color: "text-gold", bg: "bg-gold/10" };
  return { text: "NORMAL", color: "text-ink-tertiary", bg: "bg-ink-faint/10" };
}

function barColor(item: ReminderItem): string {
  if (isOverdue(item)) return "bg-type-error";
  if ((item.priority ?? 0) >= 8) return "bg-type-error";
  if ((item.priority ?? 0) >= 5) return "bg-gold";
  return "bg-type-decision";
}

// ─── Component ──────────────────────────────────────────────

interface Props {
  onCountChange?: (count: number) => void;
}

export default function Reminders({ onCountChange }: Props) {
  const [activeItems, setActiveItems] = useState<ReminderItem[]>([]);
  const [completedItems, setCompletedItems] = useState<ReminderItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actingId, setActingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch("/api/memories?event_type=reminder&limit=100");
      if (!res.ok) throw new Error("Failed to load reminders");
      const data = await res.json();
      const all: ReminderItem[] = data.memories || [];

      const active = all.filter((r) => {
        const status = r.metadata?.status;
        if (status === "completed" || status === "dismissed") return false;
        // Hide snoozed items that aren't due yet
        const snoozed = r.metadata?.snoozed_until;
        if (snoozed && new Date(snoozed).getTime() > Date.now()) return false;
        return true;
      });
      // Sort: overdue first, then by due date, then by priority
      active.sort((a, b) => {
        const aOverdue = isOverdue(a) ? 0 : 1;
        const bOverdue = isOverdue(b) ? 0 : 1;
        if (aOverdue !== bOverdue) return aOverdue - bOverdue;
        const aDue = a.metadata?.due_date ? new Date(a.metadata.due_date).getTime() : Infinity;
        const bDue = b.metadata?.due_date ? new Date(b.metadata.due_date).getTime() : Infinity;
        if (aDue !== bDue) return aDue - bDue;
        return (b.priority ?? 0) - (a.priority ?? 0);
      });
      setActiveItems(active);

      const done = all.filter((r) => {
        const status = r.metadata?.status;
        return status === "completed" || status === "dismissed";
      });
      setCompletedItems(done);

      if (onCountChange) {
        const overdueCount = active.filter(isOverdue).length;
        onCountChange(overdueCount);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Load failed");
    } finally {
      setLoading(false);
    }
  }, [onCountChange]);

  useEffect(() => {
    setLoading(true);
    load();
  }, [load]);

  async function handleAction(id: string, action: "complete" | "dismiss" | "snooze", snoozeDays?: number) {
    setActingId(id);
    setError(null);
    try {
      const item = activeItems.find((i) => i.id === id);
      if (!item) return;

      const metadata = { ...(item.metadata || {}) };

      if (action === "complete") {
        metadata.status = "completed";
        metadata.completed_at = new Date().toISOString();
      } else if (action === "dismiss") {
        metadata.status = "dismissed";
      } else if (action === "snooze" && snoozeDays) {
        const snoozeDate = new Date();
        snoozeDate.setDate(snoozeDate.getDate() + snoozeDays);
        metadata.snoozed_until = snoozeDate.toISOString();
      }

      const res = await fetch("/api/memories", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, metadata }),
      });
      if (!res.ok) throw new Error("Action failed");

      // Move item from active to completed (or just remove if snoozed)
      const updatedItem = { ...item, metadata };
      setActiveItems((prev) => {
        const next = prev.filter((i) => i.id !== id);
        if (onCountChange) {
          onCountChange(next.filter(isOverdue).length);
        }
        return next;
      });
      if (action === "complete" || action === "dismiss") {
        setCompletedItems((prev) => [updatedItem, ...prev]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setActingId(null);
    }
  }

  // ─── Loading state ─────────────────────────────────────────

  if (loading) {
    return (
      <div role="status" aria-live="polite" aria-label="Loading reminders">
        <div className="flex items-center justify-between px-5 pt-4 pb-3">
          <div className="skeleton h-4 w-24 rounded" />
        </div>
        <div className="space-y-2 px-5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="rounded-lg bg-surface border border-edge overflow-hidden relative" style={{ animationDelay: `${i * 120}ms` }}>
              <div className="h-[3px] skeleton" />
              <div className="p-3 space-y-2.5">
                <div className="flex items-center gap-2">
                  <div className="skeleton h-4 w-16 rounded" />
                  <div className="skeleton h-3 w-12 rounded" />
                </div>
                <div className="skeleton h-4 w-4/5 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const totalCount = activeItems.length + completedItems.length;

  return (
    <div className="pb-4">
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3">
        <h3 className="text-[14px] font-medium text-ink-secondary">Reminders</h3>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-5 mb-3 p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[15px] text-type-error" role="alert">
          {error}
        </div>
      )}

      {/* Empty state */}
      {totalCount === 0 && (
        <div className="px-5 py-8 text-center text-[14px] text-ink-tertiary">
          No reminders
        </div>
      )}

      {/* Active reminder cards */}
      <div className="space-y-2 px-5 admin-stagger">
        {activeItems.map((item, idx) => {
          const busy = actingId === item.id;
          const overdue = isOverdue(item);
          const badge = priorityBadge(item.priority ?? 0);
          const bar = barColor(item);
          const dueDate = item.metadata?.due_date;

          return (
            <div
              key={item.id}
              className={`rounded-lg bg-surface border overflow-hidden card-enter relative ${
                overdue ? "border-type-error/30" : "border-edge"
              }`}
              style={{ animationDelay: `${idx * 40}ms` }}
            >
              {/* Color bar */}
              <div className={`absolute top-0 left-0 right-0 h-[3px] ${bar}`} />

              <div className="p-3 pt-4">
                {/* Header row */}
                <div className="flex items-center gap-2 mb-1.5">
                  {overdue && (
                    <span className="text-[11px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wider text-type-error bg-type-error/10">
                      OVERDUE
                    </span>
                  )}
                  {!overdue && (
                    <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wider ${badge.color} ${badge.bg}`}>
                      {badge.text}
                    </span>
                  )}
                  {dueDate && (
                    <span className={`text-[13px] font-mono ${overdue ? "text-type-error font-medium" : "text-ink-faint"}`}>
                      {formatDueDate(dueDate)}
                    </span>
                  )}
                  {item.project && (
                    <span className="text-[12px] text-ink-faint/60 ml-auto">{item.project}</span>
                  )}
                </div>

                {/* Content */}
                <p className="text-[15px] text-ink leading-relaxed mb-3">
                  {item.content}
                </p>

                {/* Meta */}
                <div className="flex items-center gap-2 mb-3 text-[13px] text-ink-faint">
                  <span className="font-mono">Created {timeAgo(item.created_at)}</span>
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={() => handleAction(item.id, "complete")}
                    disabled={busy}
                    className="px-3.5 py-2 rounded-lg bg-type-lesson hover:bg-type-lesson/80 text-canvas text-[14px] font-semibold transition-all disabled:opacity-50 touch-manipulation"
                  >
                    {busy ? "..." : "Complete"}
                  </button>
                  <button
                    onClick={() => handleAction(item.id, "snooze", 1)}
                    disabled={busy}
                    className="px-3 py-2 rounded-lg border border-edge hover:bg-surface-hover text-ink-secondary text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation"
                  >
                    1d
                  </button>
                  <button
                    onClick={() => handleAction(item.id, "snooze", 3)}
                    disabled={busy}
                    className="px-3 py-2 rounded-lg border border-edge hover:bg-surface-hover text-ink-secondary text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation"
                  >
                    3d
                  </button>
                  <button
                    onClick={() => handleAction(item.id, "snooze", 7)}
                    disabled={busy}
                    className="px-3 py-2 rounded-lg border border-edge hover:bg-surface-hover text-ink-secondary text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation"
                  >
                    7d
                  </button>
                  <button
                    onClick={() => handleAction(item.id, "dismiss")}
                    disabled={busy}
                    className="px-3.5 py-2 rounded-lg border border-edge hover:bg-surface-hover text-ink-tertiary text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation ml-auto"
                  >
                    {busy ? "..." : "Dismiss"}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Separator between active and completed */}
      {activeItems.length > 0 && completedItems.length > 0 && (
        <div className="px-5 py-3">
          <div className="border-t border-edge" />
          <p className="text-[12px] text-ink-faint mt-2 uppercase tracking-wider font-medium">Completed</p>
        </div>
      )}

      {/* Completed/dismissed reminder cards (muted) */}
      <div className="space-y-2 px-5">
        {completedItems.map((item, idx) => {
          const badge = priorityBadge(item.priority ?? 0);
          const dueDate = item.metadata?.due_date;

          return (
            <div
              key={item.id}
              className="opacity-60 rounded-lg bg-surface border border-edge overflow-hidden card-enter relative"
              style={{ animationDelay: `${(activeItems.length + idx) * 40}ms` }}
            >
              {/* Color bar */}
              <div className="absolute top-0 left-0 right-0 h-[3px] bg-ink-faint/30" />

              <div className="p-3 pt-4">
                {/* Header row */}
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wider ${badge.color} ${badge.bg}`}>
                    {badge.text}
                  </span>
                  {dueDate && (
                    <span className="text-[13px] font-mono text-ink-faint">
                      {formatDueDate(dueDate)}
                    </span>
                  )}
                  {item.project && (
                    <span className="text-[12px] text-ink-faint/60 ml-auto">{item.project}</span>
                  )}
                </div>

                {/* Content */}
                <p className="text-[15px] text-ink leading-relaxed mb-3">
                  {item.content}
                </p>

                {/* Completed status */}
                <div className="flex items-center gap-2 text-[13px]">
                  <span className={`font-medium ${
                    item.metadata?.status === "completed" ? "text-type-lesson" : "text-ink-tertiary"
                  }`}>
                    {item.metadata?.status === "completed" ? "Completed" : "Dismissed"}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

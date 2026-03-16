import { useCallback, useEffect, useMemo, useState } from "react";
import { type Schedule, computeStatus } from "./jobUtils";
import { JobRow } from "./JobRow";
import { JobDetailPanel } from "./JobDetailPanel";
import ApprovalsQueue from "./ApprovalsQueue";
import UpcomingTriggers from "../insights/UpcomingTriggers";
import { useKeyboardNav } from "../hooks/useKeyboardNav";
import { useBulkSelection } from "../hooks/useBulkSelection";
import BulkActionBar from "../shared/BulkActionBar";

// ─── Job Categories ─────────────────────────────────

interface JobCategory {
  key: string;
  label: string;
  description: string;
  cls: string; // color class for the label
}

const CATEGORIES: JobCategory[] = [
  { key: "vercel_cron", label: "Vercel Crons", description: "Lightweight automation on Vercel Pro", cls: "text-[#4da3ff]" },
  { key: "github_actions", label: "GitHub Actions", description: "AI-heavy generation via GitHub", cls: "text-[#58a65c]" },
  { key: "maintenance", label: "System Maintenance", description: "Session-triggered pipeline stages", cls: "text-type-observation" },
  { key: "cowork", label: "Cowork", description: "Claude Cowork scheduled tasks", cls: "text-type-lesson" },
  { key: "other", label: "Other", description: "Uncategorized jobs", cls: "text-ink-tertiary" },
];

function getSource(s: Schedule): string {
  const src = (s.metadata as Record<string, unknown>)?.source;
  if (typeof src === "string" && ["vercel_cron", "github_actions", "maintenance", "cowork"].includes(src)) return src;
  // Fall back to label prefix for jobs created outside the seed script
  if (s.label.startsWith("com.omega.vercel")) return "vercel_cron";
  if (s.label.startsWith("com.omega.github")) return "github_actions";
  if (s.label.startsWith("com.omega.maintenance")) return "maintenance";
  if (s.label.startsWith("com.omega.cowork")) return "cowork";
  return "other";
}

export default function JobList() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showIssuesOnly, setShowIssuesOnly] = useState(false);
  const [healthMap, setHealthMap] = useState<Record<string, string[]>>({});
  const [triggersOpen, setTriggersOpen] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());

  // ─── Data Fetching ──────────────────────────────────
  // Fetch schedules + health atomically to avoid a flash where computeStatus
  // falls back to the stale last_status column before health data arrives.

  const fetchSchedules = useCallback(async () => {
    setError(null);
    try {
      const [schedRes, healthRes] = await Promise.all([
        fetch("/api/schedules"),
        fetch("/api/admin/schedule-runs/health").catch(() => null),
      ]);
      if (!schedRes.ok) throw new Error(`${schedRes.status}`);
      const json = await schedRes.json();
      const health: Record<string, string[]> = healthRes?.ok
        ? (await healthRes.json()).health ?? {}
        : {};
      setSchedules(json.schedules || []);
      setHealthMap(health);
    } catch {
      setError("Couldn't load schedules. Check your connection.");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchSchedules();
    const id = setInterval(fetchSchedules, 60_000);
    return () => clearInterval(id);
  }, [fetchSchedules]);

  // ─── Handlers ───────────────────────────────────────

  const handleUpdate = async (id: string, fields: Record<string, unknown>) => {
    try {
      const res = await fetch("/api/schedules", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, ...fields }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      await fetchSchedules();
    } catch {
      setError("Failed to update schedule.");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const res = await fetch("/api/schedules", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      // Backend only disables — update local state to reflect disabled status
      setSchedules((prev) =>
        prev.map((s) => (s.id === id ? { ...s, enabled: false } : s)),
      );
      setSelectedId(null);
    } catch {
      setError("Failed to disable schedule.");
    }
  };

  const handleToggle = async (id: string, enabled: boolean) => {
    await handleUpdate(id, { enabled });
  };

  const handleSelect = (id: string) => {
    setSelectedId(selectedId === id ? null : id);
  };

  const handleSkipToggle = useCallback(async (id: string, skipNext: boolean) => {
    try {
      const res = await fetch("/api/schedules", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, skip_next: skipNext }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      await fetchSchedules();
    } catch {
      setError("Failed to update skip status.");
    }
  }, [fetchSchedules]);

  const toggleGroup = useCallback((key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // ─── Derived State ──────────────────────────────────

  const issueCount = schedules.filter((s) => {
    const st = computeStatus(s, healthMap[s.label]);
    return st === "error" || st === "late";
  }).length;

  const displayed = showIssuesOnly
    ? schedules.filter((s) => {
        const st = computeStatus(s, healthMap[s.label]);
        return st === "error" || st === "late";
      })
    : schedules;

  const activeCount = schedules.filter((s) => s.enabled).length;

  const selectedSchedule = schedules.find((s) => s.id === selectedId) ?? null;

  // Group displayed jobs by source category
  const grouped = useMemo(() => {
    const map = new Map<string, Schedule[]>();
    for (const s of displayed) {
      const src = getSource(s);
      if (!map.has(src)) map.set(src, []);
      map.get(src)!.push(s);
    }
    // Return in CATEGORIES order, skipping empty groups
    return CATEGORIES
      .filter((c) => map.has(c.key))
      .map((c) => ({ ...c, jobs: map.get(c.key)! }));
  }, [displayed]);

  // ─── Keyboard Nav & Bulk Selection ─────────────────

  const scheduleIds = useMemo(() => (schedules ?? []).map((s: Schedule) => s.id), [schedules]);

  const { focusedId } = useKeyboardNav({
    dataAttribute: "data-job-id",
    ids: scheduleIds,
    onSelect: (id) => handleSelect(id),
    extraKeys: {
      t: (id) => handleToggle(id, !(schedules ?? []).find((s: Schedule) => s.id === id)?.enabled),
    },
  });

  const bulk = useBulkSelection(scheduleIds);

  const handleBulkToggle = async (enabled: boolean) => {
    await Promise.all([...bulk.selectedIds].map((id) => handleToggle(id, enabled)));
    bulk.clear();
  };

  // ─── Loading State ──────────────────────────────────

  if (loading) {
    return (
      <div role="status" aria-live="polite" aria-label="Loading schedules">
        <div className="px-5 pt-6 pb-2">
          <div className="skeleton h-4 w-24 rounded mb-4" />
        </div>
        <div className="px-5 space-y-0">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="py-3 border-b border-white/[0.04]"
              style={{ animationDelay: `${i * 120}ms` }}
            >
              <div className="flex items-center gap-4">
                <div className="skeleton h-5 w-16 rounded-full" />
                <div className="flex-1 space-y-1.5">
                  <div className="skeleton h-4 w-2/5 rounded" />
                  <div className="skeleton h-3 w-3/5 rounded" />
                </div>
                <div className="skeleton h-4 w-24 rounded" />
                <div className="skeleton h-4 w-12 rounded" />
                <div className="skeleton h-4 w-12 rounded" />
                <div className="skeleton h-5 w-9 rounded-full" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ─── Content ────────────────────────────────────────

  return (
    <div className="pb-4">
      {/* Header */}
      <div className="px-5 pt-5 pb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="admin-section-label text-gold/80">Scheduled Jobs</h2>
          <span className="text-[14px] text-ink-faint tabular-nums">
            {activeCount} active
          </span>
          <div className="flex-1 h-px bg-gradient-to-r from-gold/10 to-transparent ml-1 min-w-[40px]" />
        </div>

        <div className="flex items-center gap-2">
          {/* Issues filter */}
          {issueCount > 0 && (
            <button
              onClick={() => setShowIssuesOnly(!showIssuesOnly)}
              className={`text-[14px] font-medium px-3 py-1.5 rounded-full transition-colors touch-manipulation flex items-center gap-1.5 min-h-[44px] ${
                showIssuesOnly
                  ? "bg-type-error/[0.12] text-type-error"
                  : "bg-surface-elevated text-ink-tertiary hover:text-type-error"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-type-error" />
              Issues ({issueCount})
            </button>
          )}

          {/* Refresh */}
          <button
            onClick={() => {
              setLoading(true);
              fetchSchedules();
            }}
            className="p-2.5 rounded-lg text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated transition-all touch-manipulation min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Refresh schedules"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Pending approvals queue */}
      <ApprovalsQueue />

      {/* Upcoming triggers (collapsible) */}
      {schedules.length > 0 && (
        <div className="mx-5 mb-3">
          <button
            onClick={() => setTriggersOpen(!triggersOpen)}
            className="flex items-center gap-2 text-[13px] text-ink-tertiary hover:text-ink-secondary transition-colors mb-2"
          >
            <svg
              className={`w-3.5 h-3.5 transition-transform ${triggersOpen ? "rotate-90" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
            </svg>
            Upcoming Triggers
          </button>
          {triggersOpen && (
            <UpcomingTriggers schedules={schedules} onSkipToggle={handleSkipToggle} />
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="mx-5 mb-3 p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[16px] text-type-error"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* Empty state */}
      {schedules.length === 0 ? (
        <div className="text-center px-8 py-16">
          <div className="relative w-16 h-16 mx-auto mb-5">
            <div className="absolute inset-0 flex items-center justify-center">
              <svg
                className="w-8 h-8 text-gold/20"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5"
                />
              </svg>
            </div>
          </div>
          <p className="text-[16px] text-ink-tertiary leading-relaxed">
            No scheduled jobs found. Run the seed script to import your launchd
            plists.
          </p>
        </div>
      ) : (
        <>
          {/* Grouped job sections */}
          {grouped.map((group) => {
            const isCollapsed = collapsedGroups.has(group.key);
            const groupActive = group.jobs.filter((s) => s.enabled).length;
            const groupIssues = group.jobs.filter((s) => {
              const st = computeStatus(s, healthMap[s.label]);
              return st === "error" || st === "late";
            }).length;

            return (
              <div key={group.key} className="mb-1">
                {/* Section header */}
                <button
                  type="button"
                  onClick={() => toggleGroup(group.key)}
                  className="w-full px-5 py-2.5 flex items-center gap-2 hover:bg-white/[0.02] transition-colors"
                >
                  <svg
                    className={`w-3 h-3 text-ink-faint transition-transform ${isCollapsed ? "" : "rotate-90"}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2.5}
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                  </svg>
                  <span className={`text-[12px] font-semibold uppercase tracking-wider ${group.cls}`}>
                    {group.label}
                  </span>
                  <span className="text-[11px] text-ink-faint tabular-nums">
                    {groupActive} active
                  </span>
                  {groupIssues > 0 && (
                    <span className="min-w-[16px] h-[16px] bg-type-error/20 text-type-error text-[10px] font-bold rounded-full flex items-center justify-center px-1">
                      {groupIssues}
                    </span>
                  )}
                  <div className="flex-1 h-px bg-white/[0.04] ml-1" />
                </button>

                {/* Column headers + rows (collapsible) */}
                {!isCollapsed && (
                  <>
                    <div className="px-5 pb-1 pt-1">
                      <div className="grid grid-cols-[84px_56px_1fr_140px_72px_72px_44px] items-center gap-3 text-[12px] text-ink-faint">
                        <span>Status</span>
                        <span>Health</span>
                        <span>Name</span>
                        <span>Schedule</span>
                        <span>Last run</span>
                        <span>Next run</span>
                        <span />
                      </div>
                    </div>
                    <div>
                      {group.jobs.map((s) => (
                        <JobRow
                          key={s.id}
                          schedule={s}
                          selected={s.id === selectedId}
                          onSelect={handleSelect}
                          onToggle={handleToggle}
                          healthDots={healthMap[s.label]}
                          isFocused={focusedId === s.id}
                          isChecked={bulk.isSelected(s.id)}
                        />
                      ))}
                    </div>
                  </>
                )}
              </div>
            );
          })}

          {showIssuesOnly && displayed.length === 0 && (
            <div className="text-center py-8 text-[13px] text-ink-tertiary">
              No issues found. All jobs healthy.
            </div>
          )}
        </>
      )}

      {/* Detail panel */}
      {selectedSchedule && (
        <JobDetailPanel
          schedule={selectedSchedule}
          onClose={() => setSelectedId(null)}
          onUpdate={handleUpdate}
          onDelete={handleDelete}
          healthDots={healthMap[selectedSchedule.label]}
        />
      )}

      {/* Bulk action bar */}
      <BulkActionBar count={bulk.count} onClear={bulk.clear}>
        <button onClick={() => handleBulkToggle(true)} className="px-3 py-1.5 text-[13px] font-medium rounded-lg bg-type-lesson/10 text-type-lesson hover:bg-type-lesson/20 transition-colors">
          Enable
        </button>
        <button onClick={() => handleBulkToggle(false)} className="px-3 py-1.5 text-[13px] font-medium rounded-lg bg-type-error/10 text-type-error hover:bg-type-error/20 transition-colors">
          Disable
        </button>
      </BulkActionBar>
    </div>
  );
}

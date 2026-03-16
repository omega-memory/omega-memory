import { useState, useCallback } from "react";
import type { AlertContext, AlertType, AlertAction, Tab, FailingJobContext, FailedPostContext, CloudSyncGapContext, MemorySpikeContext, CoordinationConflictContext } from "../lib/types";

// ─── Map problem text to alert type + params ────────────────

interface AlertMapping {
  type: AlertType;
  params?: Record<string, string>;
  actions: AlertAction[];
}

export function mapProblemToAlert(text: string): AlertMapping | null {
  if (text.match(/job[s]? failing/i)) {
    // Extract job names from "2 jobs failing: job1, job2"
    const match = text.match(/failing:\s*(.+)/);
    const jobName = match?.[1]?.split(",")[0]?.trim() ?? "";
    return {
      type: "failing_job",
      params: { job: jobName },
      actions: [
        { type: "retry_job", label: "Retry", jobLabel: jobName, variant: "primary" },
        { type: "navigate", label: "View Jobs", tab: "jobs" as Tab, variant: "secondary" },
      ],
    };
  }
  if (text.match(/post[s]? failed/i)) {
    return {
      type: "failed_post",
      actions: [
        { type: "requeue_post", label: "Re-queue", variant: "primary" },
        { type: "navigate", label: "View Actions", tab: "actions" as Tab, variant: "secondary" },
      ],
    };
  }
  if (text.match(/cloud sync gap/i) || text.match(/cloud sync has no data/i)) {
    return {
      type: "cloud_sync_gap",
      actions: [
        { type: "force_sync", label: "Force Sync", variant: "primary" },
        { type: "dismiss", label: "Dismiss", variant: "secondary" },
      ],
    };
  }
  if (text.match(/engagement.*declining/i)) {
    return {
      type: "engagement_declining",
      actions: [
        { type: "navigate", label: "View Insights", tab: "insights" as Tab, variant: "primary" },
        { type: "dismiss", label: "Dismiss", variant: "secondary" },
      ],
    };
  }
  if (text.match(/overdue/i)) {
    return {
      type: "overdue_job",
      actions: [
        { type: "navigate", label: "View Jobs", tab: "jobs" as Tab, variant: "secondary" },
      ],
    };
  }
  return null;
}

// ─── Action button styling ──────────────────────────────────

const ACTION_STYLES = {
  primary: "bg-gold/10 text-gold border-gold/20 hover:bg-gold/20",
  secondary: "bg-surface-elevated text-ink-secondary border-edge hover:bg-surface-hover",
  danger: "bg-type-error/10 text-type-error border-type-error/20 hover:bg-type-error/20",
};

// ─── Alert Context Detail Renderers ─────────────────────────

function FailingJobDetail({ detail }: { detail: FailingJobContext }) {
  return (
    <div className="space-y-3">
      {detail.detail.lastError && (
        <div className="px-3 py-2 rounded-lg bg-type-error/[0.05] border border-type-error/10">
          <span className="text-[11px] font-mono text-type-error/60 uppercase tracking-wider">Last Error</span>
          <p className="text-[13px] text-ink-secondary mt-1 font-mono break-all">{detail.detail.lastError}</p>
        </div>
      )}
      <div>
        <span className="text-[11px] font-mono text-ink-faint uppercase tracking-wider">Recent Runs</span>
        <div className="mt-1.5 space-y-1">
          {detail.detail.recentRuns.map((run, i) => (
            <div key={i} className="flex items-center gap-3 text-[12px]">
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${run.status === "error" ? "bg-type-error" : run.status === "ok" ? "bg-type-lesson" : "bg-ink-faint"}`} />
              <span className="text-ink-secondary flex-1">{run.status}</span>
              <span className="text-ink-faint tabular-nums">{run.durationMs ? `${(run.durationMs / 1000).toFixed(1)}s` : "-"}</span>
              <span className="text-ink-faint tabular-nums">{new Date(run.startedAt).toLocaleString()}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FailedPostDetail({ detail }: { detail: FailedPostContext }) {
  return (
    <div className="space-y-3">
      {detail.detail.recentFailed.length > 0 && (
        <div>
          <span className="text-[11px] font-mono text-type-error/60 uppercase tracking-wider">Failed</span>
          <div className="mt-1.5 space-y-2">
            {detail.detail.recentFailed.map((f, i) => (
              <div key={i} className="px-3 py-2 rounded-lg bg-type-error/[0.03] border border-type-error/[0.06]">
                <p className="text-[13px] text-ink-secondary line-clamp-2">{f.content}</p>
                {f.reason && <p className="text-[11px] text-type-error/70 mt-1">Reason: {f.reason}</p>}
                <p className="text-[11px] text-ink-faint mt-1">@{f.account}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      {detail.detail.recentSuccessful.length > 0 && (
        <div>
          <span className="text-[11px] font-mono text-type-lesson/60 uppercase tracking-wider">Recent Successful</span>
          <div className="mt-1.5 space-y-1">
            {detail.detail.recentSuccessful.map((s, i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="w-1.5 h-1.5 rounded-full bg-type-lesson shrink-0 mt-1.5" />
                <span className="text-ink-faint line-clamp-1 flex-1">{s.content}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CloudSyncDetail({ detail }: { detail: CloudSyncGapContext }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <div className="text-[18px] font-light text-ink tabular-nums">{detail.detail.localCount.toLocaleString()}</div>
          <div className="text-[11px] text-ink-faint uppercase">Local</div>
        </div>
        <div className="text-center">
          <div className="text-[18px] font-light text-ink tabular-nums">{detail.detail.cloudCount.toLocaleString()}</div>
          <div className="text-[11px] text-ink-faint uppercase">Cloud</div>
        </div>
        <div className="text-center">
          <div className="text-[18px] font-light text-gold tabular-nums">{detail.detail.unsyncedCount.toLocaleString()}</div>
          <div className="text-[11px] text-ink-faint uppercase">Unsynced</div>
        </div>
      </div>
      {detail.detail.lastSyncAt && (
        <p className="text-[12px] text-ink-faint text-center">Last sync: {new Date(detail.detail.lastSyncAt).toLocaleString()}</p>
      )}
    </div>
  );
}

function MemorySpikeDetail({ detail }: { detail: MemorySpikeContext }) {
  return (
    <div>
      <span className="text-[11px] font-mono text-ink-faint uppercase tracking-wider">{detail.detail.totalInLastHour} memories in last hour</span>
      <div className="mt-1.5 space-y-1">
        {detail.detail.recentMemories.map((m, i) => (
          <div key={i} className="flex items-start gap-2 text-[12px]">
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-surface-elevated text-ink-faint border border-edge shrink-0">{m.memoryType}</span>
            <span className="text-ink-secondary line-clamp-1 flex-1">{m.content}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CoordinationConflictDetail({ detail }: { detail: CoordinationConflictContext }) {
  return (
    <div className="space-y-2">
      {detail.detail.conflicts.map((c, i) => (
        <div key={i} className="px-3 py-2 rounded-lg bg-type-error/[0.03] border border-type-error/[0.06]">
          <p className="text-[13px] text-ink font-mono">{c.filePath.split("/").slice(-2).join("/")}</p>
          <div className="mt-1 flex flex-wrap gap-2">
            {c.sessions.map((s, j) => (
              <span key={j} className="text-[11px] px-2 py-0.5 rounded-full bg-surface-elevated border border-edge text-ink-secondary">
                {s.sessionId.slice(0, 8)} ({s.project})
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function AlertContextDetail({ context }: { context: AlertContext }) {
  switch (context.type) {
    case "failing_job": return <FailingJobDetail detail={context as FailingJobContext} />;
    case "failed_post": return <FailedPostDetail detail={context as FailedPostContext} />;
    case "cloud_sync_gap": return <CloudSyncDetail detail={context as CloudSyncGapContext} />;
    case "engagement_declining": return null; // Simple text is enough
    case "memory_spike": return <MemorySpikeDetail detail={context as MemorySpikeContext} />;
    case "coordination_conflict": return <CoordinationConflictDetail detail={context as CoordinationConflictContext} />;
    default: return null;
  }
}

// ─── Main AlertCard Component ───────────────────────────────

interface AlertCardProps {
  text: string;
  urgent: boolean;
  mapping: AlertMapping;
  onNavigate: (tab: Tab) => void;
}

export default function AlertCard({ text, urgent, mapping, onNavigate }: AlertCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [context, setContext] = useState<AlertContext | null>(null);
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchContext = useCallback(async () => {
    if (context) return; // Already loaded
    setLoading(true);
    try {
      const params = new URLSearchParams({ type: mapping.type, ...mapping.params });
      const res = await fetch(`/api/admin/alert-context?${params}`);
      if (res.ok) {
        setContext(await res.json());
      }
    } catch {
      // Non-critical
    } finally {
      setLoading(false);
    }
  }, [context, mapping]);

  const handleToggle = useCallback(() => {
    const next = !expanded;
    setExpanded(next);
    if (next) fetchContext();
  }, [expanded, fetchContext]);

  const handleAction = useCallback(async (action: AlertAction) => {
    setActionLoading(action.type);
    try {
      switch (action.type) {
        case "navigate":
          if (action.tab) onNavigate(action.tab);
          break;
        case "dismiss":
          setDismissed(true);
          break;
        case "snooze": {
          setDismissed(true);
          // Store snooze in localStorage
          const snoozed = JSON.parse(localStorage.getItem("admin_snoozed_alerts") ?? "{}");
          snoozed[mapping.type] = Date.now() + 60 * 60_000; // 1 hour
          localStorage.setItem("admin_snoozed_alerts", JSON.stringify(snoozed));
          break;
        }
        case "retry_job":
          if (action.jobLabel) {
            await fetch("/api/schedules/trigger", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ label: action.jobLabel }),
            });
          }
          break;
        case "force_sync":
          await fetch("/api/schedules/trigger", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: "cloud-sync" }),
          });
          break;
        case "requeue_post":
          // Navigate to actions tab where re-queue is handled
          onNavigate("actions" as Tab);
          break;
      }
    } catch {
      // Silently fail
    } finally {
      setActionLoading(null);
    }
  }, [mapping, onNavigate]);

  if (dismissed) return null;

  return (
    <div className={`rounded-lg transition-all duration-200 ${urgent ? "bg-gold/[0.03]" : ""}`}>
      {/* Header row */}
      <button
        onClick={handleToggle}
        className="flex items-center gap-3 px-3 py-2 w-full text-left cursor-pointer group"
      >
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${urgent ? "bg-gold/60" : "bg-ink-faint/20"}`} />
        <span className={`text-[14px] leading-snug flex-1 ${urgent ? "text-ink font-medium" : "text-ink-faint"}`}>{text}</span>
        <svg
          className={`w-3.5 h-3.5 text-ink-faint transition-transform duration-150 ${expanded ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {/* Expandable context + actions */}
      <div className="grid transition-[grid-template-rows] duration-300 ease-out" style={{ gridTemplateRows: expanded ? "1fr" : "0fr" }}>
        <div className="overflow-hidden">
          <div className="px-3 pb-3 space-y-3">
            <div className="h-px bg-edge-subtle ml-4" />

            {/* Loading state */}
            {loading && (
              <div className="flex items-center gap-2 px-3 py-2">
                <svg className="w-3.5 h-3.5 animate-spin text-ink-faint" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
                </svg>
                <span className="text-[12px] text-ink-faint">Loading context...</span>
              </div>
            )}

            {/* Context detail */}
            {context && <AlertContextDetail context={context} />}

            {/* Action buttons */}
            <div className="flex items-center gap-2 ml-4">
              {mapping.actions.map((action) => (
                <button
                  key={action.type}
                  onClick={() => handleAction(action)}
                  disabled={actionLoading === action.type}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-colors disabled:opacity-50 ${ACTION_STYLES[action.variant ?? "secondary"]}`}
                >
                  {actionLoading === action.type && (
                    <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
                    </svg>
                  )}
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

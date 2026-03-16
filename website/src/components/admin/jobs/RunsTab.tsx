import { useCallback, useEffect, useState } from "react";

import { type ScheduleRun, formatCost, formatTokens, formatDuration } from "./jobUtils";

export function RunsTab({ scheduleLabel }: { scheduleLabel: string }) {
  const [runs, setRuns] = useState<ScheduleRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedError, setExpandedError] = useState<string | null>(null);

  const fetchRuns = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/admin/schedule-runs?label=${encodeURIComponent(scheduleLabel)}&limit=20`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setRuns(data.runs ?? []))
      .catch(() => setError("Failed to load run history"))
      .finally(() => setLoading(false));
  }, [scheduleLabel]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  if (loading) {
    return (
      <div className="space-y-3 pt-2">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-12 rounded-lg skeleton" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-10">
        <p className="text-[15px] text-type-error">{error}</p>
        <button
          onClick={fetchRuns}
          className="mt-3 text-[14px] text-ink-tertiary hover:text-ink-secondary underline transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="text-center py-10 text-[15px] text-ink-tertiary">
        No runs recorded yet. Runs will appear after the next execution.
      </div>
    );
  }

  return (
    <div className="space-y-1 pt-2">
      {/* Header */}
      <div className="grid grid-cols-[60px_1fr_56px_56px_64px] gap-2 px-2 text-[12px] text-ink-faint uppercase tracking-wider">
        <span>Status</span>
        <span>Started</span>
        <span>Time</span>
        <span>Tokens</span>
        <span className="text-right">Cost</span>
      </div>

      {runs.map((run) => {
        const isError = run.status === "error";
        const isExpanded = expandedError === run.id;
        return (
          <div key={run.id}>
            <button
              onClick={() => isError ? setExpandedError(isExpanded ? null : run.id) : undefined}
              className={`w-full grid grid-cols-[60px_1fr_56px_56px_64px] gap-2 px-2 py-2 rounded-lg text-left transition-colors ${
                isError ? "hover:bg-type-error/[0.04] cursor-pointer" : ""
              } ${isExpanded ? "bg-type-error/[0.04]" : ""}`}
            >
              <span className={`text-[13px] font-medium ${
                run.status === "ok" ? "text-type-lesson" :
                run.status === "error" ? "text-type-error" :
                run.status === "dead_letter" ? "text-type-error" :
                run.status === "pending_approval" ? "text-type-reminder" :
                run.status === "approved" ? "text-type-lesson" :
                run.status === "rejected" ? "text-type-error" :
                "text-ink-tertiary"
              }`}>
                {run.status === "running" ? "..." :
                 run.status === "dead_letter" ? "DLQ" :
                 run.status === "pending_approval" ? "pending" :
                 run.status}
              </span>
              <span className="text-[14px] text-ink-secondary truncate">
                {new Date(run.started_at).toLocaleString("en-US", {
                  month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
                })}
              </span>
              <span className="text-[14px] text-ink-tertiary font-mono tabular-nums">
                {formatDuration(run.duration_ms)}
              </span>
              <span className="text-[14px] text-ink-tertiary font-mono tabular-nums">
                {formatTokens(run.input_tokens + run.output_tokens)}
              </span>
              <span className="text-[14px] text-ink-tertiary font-mono tabular-nums text-right">
                {formatCost(Number(run.cost_usd))}
              </span>
            </button>
            {isExpanded && run.error && (
              <div className="mx-2 mb-2 p-3 rounded-lg bg-type-error/[0.06] border border-type-error/10 text-[13px] text-type-error/80 font-mono break-all">
                {run.error}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";

import { relativeTime } from "./jobUtils";

// ─── Types ──────────────────────────────────────────

interface AuditEvent {
  id: string;
  run_id: string | null;
  schedule_label: string;
  event_type: string;
  actor: string;
  actor_ip: string | null;
  details: Record<string, unknown> | null;
  parent_event_id: string | null;
  created_at: string;
}

// ─── Event Type Badge Config ────────────────────────

type BadgeStyle = { textCls: string; bgCls: string; dotCls: string };

const BADGE_STYLES: Record<string, BadgeStyle> = {
  // Green: success states
  completed: { textCls: "text-type-lesson", bgCls: "bg-type-lesson/[0.12]", dotCls: "bg-type-lesson" },
  approved: { textCls: "text-type-lesson", bgCls: "bg-type-lesson/[0.12]", dotCls: "bg-type-lesson" },
  // Red: failure states
  failed: { textCls: "text-type-error", bgCls: "bg-type-error/[0.12]", dotCls: "bg-type-error" },
  rejected: { textCls: "text-type-error", bgCls: "bg-type-error/[0.12]", dotCls: "bg-type-error" },
  // Blue: initiation states
  triggered: { textCls: "text-type-decision", bgCls: "bg-type-decision/[0.12]", dotCls: "bg-type-decision" },
  started: { textCls: "text-type-decision", bgCls: "bg-type-decision/[0.12]", dotCls: "bg-type-decision" },
};

const DEFAULT_BADGE: BadgeStyle = {
  textCls: "text-ink-tertiary",
  bgCls: "bg-ink-tertiary/[0.12]",
  dotCls: "bg-ink-tertiary",
};

function badgeFor(eventType: string): BadgeStyle {
  return BADGE_STYLES[eventType] ?? DEFAULT_BADGE;
}

// ─── Detail Summary ─────────────────────────────────

function summarizeDetails(details: Record<string, unknown> | null): string | null {
  if (!details) return null;

  // Show error message if present
  if (details.error && typeof details.error === "string") {
    return details.error.length > 120
      ? details.error.slice(0, 120) + "..."
      : details.error;
  }

  // Show message if present
  if (details.message && typeof details.message === "string") {
    return details.message;
  }

  // Show step name if present
  if (details.step && typeof details.step === "string") {
    return `Step: ${details.step}`;
  }

  // Fallback: count of keys
  const keys = Object.keys(details);
  if (keys.length === 0) return null;
  if (keys.length <= 3) return keys.join(", ");
  return `${keys.length} fields`;
}

// ─── Component ──────────────────────────────────────

export function AuditTab({ scheduleLabel }: { scheduleLabel: string }) {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchEvents = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(
      `/api/admin/audit-log?label=${encodeURIComponent(scheduleLabel)}&limit=50`,
    )
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setEvents(data.events ?? []))
      .catch(() => setError("Failed to load audit log"))
      .finally(() => setLoading(false));
  }, [scheduleLabel]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // ─── Loading Skeleton ───────────────────────────────

  if (loading) {
    return (
      <div className="space-y-3 pt-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="flex items-start gap-3 px-2">
            <div className="w-2 h-2 rounded-full skeleton mt-2 shrink-0" />
            <div className="flex-1 space-y-1.5">
              <div className="h-4 w-24 rounded skeleton" />
              <div className="h-3 w-48 rounded skeleton" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  // ─── Error State ────────────────────────────────────

  if (error) {
    return (
      <div className="text-center py-10">
        <p className="text-[15px] text-type-error">{error}</p>
        <button
          onClick={fetchEvents}
          className="mt-3 text-[14px] text-ink-tertiary hover:text-ink-secondary underline transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  // ─── Empty State ────────────────────────────────────

  if (events.length === 0) {
    return (
      <div className="text-center py-10 text-[15px] text-ink-tertiary">
        No audit events recorded yet. Events will appear after the next
        execution.
      </div>
    );
  }

  // ─── Timeline ───────────────────────────────────────

  return (
    <div className="space-y-0 pt-2">
      {events.map((evt, idx) => {
        const badge = badgeFor(evt.event_type);
        const detailSummary = summarizeDetails(evt.details);
        const isLast = idx === events.length - 1;
        const isExpanded = expandedId === evt.id;
        const hasDetails =
          evt.details && Object.keys(evt.details).length > 0;

        return (
          <div key={evt.id} className="flex items-stretch gap-3 px-2">
            {/* Timeline track */}
            <div className="flex flex-col items-center w-3 shrink-0">
              <div
                className={`w-2 h-2 rounded-full mt-[7px] shrink-0 ${badge.dotCls}`}
              />
              {!isLast && (
                <div className="w-px flex-1 bg-edge" />
              )}
            </div>

            {/* Event content */}
            <div className={`flex-1 pb-4 ${isLast ? "" : ""}`}>
              <button
                onClick={() =>
                  hasDetails
                    ? setExpandedId(isExpanded ? null : evt.id)
                    : undefined
                }
                className={`w-full text-left ${hasDetails ? "cursor-pointer" : "cursor-default"}`}
              >
                <div className="flex items-center gap-2 flex-wrap">
                  {/* Event type badge */}
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-[12px] font-medium ${badge.textCls} ${badge.bgCls}`}
                  >
                    {evt.event_type.replace(/_/g, " ")}
                  </span>

                  {/* Actor */}
                  <span className="text-[12px] text-ink-tertiary">
                    {evt.actor}
                  </span>

                  {/* Timestamp */}
                  <span className="text-[12px] text-ink-faint ml-auto">
                    {relativeTime(evt.created_at)}
                  </span>
                </div>

                {/* Detail summary (one-liner) */}
                {detailSummary && !isExpanded && (
                  <p className="text-[13px] text-ink-secondary mt-1 truncate">
                    {detailSummary}
                  </p>
                )}
              </button>

              {/* Expanded details */}
              {isExpanded && evt.details && (
                <div className="mt-2 p-3 rounded-lg bg-surface-elevated border border-edge text-[13px] text-ink-secondary font-mono break-all whitespace-pre-wrap">
                  {JSON.stringify(evt.details, null, 2)}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

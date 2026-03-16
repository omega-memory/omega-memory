import { useState } from "react";
import type { OrchestratorFeedItem } from "../lib/orchestrator-feed";
import ProposedChanges from "./ProposedChanges";

// ── Visual config per item_type ──────────────────────────────────

const CARD_CONFIG: Record<string, { color: string; label: string }> = {
  update:   { color: "var(--color-type-task)",     label: "Update" },
  insight:  { color: "var(--color-type-lesson)",   label: "Insight" },
  report:   { color: "var(--color-type-decision)", label: "Report" },
  alert:    { color: "var(--color-type-reminder)", label: "Alert" },
  proposal: { color: "var(--color-gold)",          label: "Proposal" },
  plan:     { color: "var(--color-gold)",          label: "Plan" },
};

const RISK_BADGE: Record<string, { bg: string; text: string; label: string }> = {
  low:    { bg: "bg-type-task/15",     text: "text-type-task",     label: "Low" },
  medium: { bg: "bg-type-reminder/15", text: "text-type-reminder", label: "Medium" },
  high:   { bg: "bg-gold/15",          text: "text-gold",          label: "High risk" },
};

// ── Icons ────────────────────────────────────────────────────────

function ItemIcon({ type }: { type: string }) {
  const cls = "w-4 h-4";
  switch (type) {
    case "update":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
        </svg>
      );
    case "insight":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 0 0-2.455 2.456Z" />
        </svg>
      );
    case "report":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
        </svg>
      );
    case "alert":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0" />
        </svg>
      );
    case "proposal":
    case "plan":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
        </svg>
      );
    default:
      return null;
  }
}

// ── Time formatting ──────────────────────────────────────────────

function formatTimeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Component ────────────────────────────────────────────────────

interface OrchestratorCardProps {
  item: OrchestratorFeedItem;
  onApprove?: (id: string, reviewNote: string) => void;
  onReject?: (id: string, reviewNote: string) => void;
  onMarkRead?: (id: string) => void;
  style?: React.CSSProperties;
}

export default function OrchestratorCard({ item, onApprove, onReject, onMarkRead, style }: OrchestratorCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [reviewNote, setReviewNote] = useState("");
  const [acting, setActing] = useState(false);

  const config = CARD_CONFIG[item.item_type] || CARD_CONFIG.update;
  const risk = RISK_BADGE[item.risk_level] || RISK_BADGE.low;
  const isHighRisk = item.risk_level === "high";
  const isActionable = item.status === "active";
  const hasBody = !!item.body;

  async function handleApprove() {
    if (!onApprove) return;
    setActing(true);
    try {
      await onApprove(item.id, reviewNote);
    } finally {
      setActing(false);
    }
  }

  async function handleReject() {
    if (!onReject) return;
    setActing(true);
    try {
      await onReject(item.id, reviewNote);
    } finally {
      setActing(false);
    }
  }

  return (
    <div className="admin-card overflow-hidden card-enter relative" style={style}>
      {/* Color bar */}
      <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ background: config.color }} />

      <div className="p-4 pt-5">
        {/* Badge row */}
        <div className="flex items-center gap-2 mb-2">
          <span style={{ color: config.color }}>
            <ItemIcon type={item.item_type} />
          </span>
          <span
            className="text-[11px] font-semibold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded"
            style={{ color: config.color, background: `color-mix(in srgb, ${config.color} 12%, transparent)` }}
          >
            {config.label}
          </span>
          {item.risk_level !== "low" && (
            <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded ${risk.bg} ${risk.text}`}>
              {risk.label}
            </span>
          )}
          {item.status !== "active" && (
            <span className="text-[11px] font-medium text-ink-faint px-1.5 py-0.5 rounded bg-surface capitalize">
              {item.status}
            </span>
          )}
          <span className="ml-auto text-[12px] text-ink-faint tabular-nums shrink-0">
            {formatTimeAgo(item.created_at)}
          </span>
        </div>

        {/* Title */}
        <h4 className="text-[15px] font-medium text-ink mb-1">{item.title}</h4>

        {/* Account */}
        {item.account && (
          <p className="text-[13px] text-ink-faint mb-2">@{item.account}</p>
        )}

        {/* Body (expandable) */}
        {hasBody && (
          <>
            <div
              className={`text-[13px] text-ink-secondary leading-relaxed whitespace-pre-wrap ${
                !expanded ? "line-clamp-3" : ""
              }`}
            >
              {item.body}
            </div>
            {item.body && item.body.length > 150 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-[13px] text-gold font-medium mt-1 touch-manipulation"
              >
                {expanded ? "Show less" : "Show more"}
              </button>
            )}
          </>
        )}

        {/* Proposed changes diff */}
        <ProposedChanges metadata={item.metadata} />

        {/* Review note input (always visible for actionable high-risk items) */}
        {isHighRisk && isActionable && (
          <textarea
            value={reviewNote}
            onChange={(e) => setReviewNote(e.target.value)}
            placeholder="Add a review note (optional)..."
            className="w-full mt-3 p-2.5 text-[13px] bg-surface border border-edge rounded-lg text-ink placeholder:text-ink-faint resize-none focus:outline-none focus:border-gold/40"
            rows={2}
          />
        )}

        {/* Actions */}
        {isActionable && (
          <div className="flex items-center gap-2.5 mt-3">
            {isHighRisk ? (
              <>
                <button
                  onClick={handleApprove}
                  disabled={acting}
                  className="px-4 py-2 rounded-lg bg-gold text-canvas text-[14px] font-semibold min-h-[36px] touch-manipulation disabled:opacity-50 transition-all"
                >
                  {acting ? "..." : "Approve"}
                </button>
                <button
                  onClick={handleReject}
                  disabled={acting}
                  className="px-4 py-2 rounded-lg border border-edge text-[14px] font-medium text-ink-secondary min-h-[36px] touch-manipulation disabled:opacity-50 transition-all hover:border-edge-strong"
                >
                  {acting ? "..." : "Reject"}
                </button>
              </>
            ) : (
              <button
                onClick={() => onMarkRead?.(item.id)}
                className="px-3 py-1.5 rounded-lg text-[13px] font-medium text-ink-tertiary hover:text-ink-secondary hover:bg-surface transition-all touch-manipulation"
              >
                Mark read
              </button>
            )}
          </div>
        )}

        {/* Review result */}
        {item.review_note && (
          <div className="mt-3 pt-3 border-t border-edge">
            <p className="text-[12px] text-ink-faint">
              {item.status === "approved" ? "Approved" : "Rejected"} {item.reviewed_at ? formatTimeAgo(item.reviewed_at) : ""}
            </p>
            <p className="text-[13px] text-ink-secondary mt-0.5">{item.review_note}</p>
          </div>
        )}

        {/* Source attribution */}
        {item.source && item.source !== "admin" && item.source !== "event_handler" && (
          <p className="mt-3 text-[11px] text-ink-faint">via {item.source}</p>
        )}
      </div>
    </div>
  );
}

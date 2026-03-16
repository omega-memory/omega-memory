import { useState, useRef, useEffect } from "react";

// ── Local types (subset of fields needed) ────────────────────────

export interface TweetItem {
  id: string;
  type: "tweet" | "reply" | "reply_alert";
  text: string;
  status: string;
  platform: "x" | "reddit" | "linkedin" | "email";
  x_account?: string;
  created_at?: string;
  scheduled_for?: string;
  source_tweet_text?: string;
  source_author_handle?: string;
}

export interface ProposalItem {
  id: string;
  item_type: "proposal" | "plan";
  risk_level: "low" | "medium" | "high";
  status: string;
  title: string;
  body: string | null;
  metadata: Record<string, unknown>;
  source: string | null;
  created_at: string;
}

interface ApprovalCardProps {
  type: "tweet" | "proposal";
  item: TweetItem | ProposalItem;
  onAction: (id: string) => void;
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

// ── Spinner ──────────────────────────────────────────────────────

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

// ── Tweet Card ───────────────────────────────────────────────────

function TweetCard({ item, onAction }: { item: TweetItem; onAction: (id: string) => void }) {
  const [acting, setActing] = useState<"approve" | "reject" | "edit" | null>(null);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(item.text);
  const [displayText, setDisplayText] = useState(item.text);
  const [dismissed, setDismissed] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  async function handleApprove() {
    setActing("approve");
    try {
      const res = await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve" }),
      });
      if (!res.ok) throw new Error("Approve failed");
      setDismissed(true);
      setTimeout(() => onAction(item.id), 300);
    } catch {
      setActing(null);
    }
  }

  async function handleReject() {
    setActing("reject");
    try {
      const res = await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reject" }),
      });
      if (!res.ok) throw new Error("Reject failed");
      setDismissed(true);
      setTimeout(() => onAction(item.id), 300);
    } catch {
      setActing(null);
    }
  }

  async function handleEditSave() {
    if (editText.trim() === displayText) {
      setEditing(false);
      return;
    }
    setActing("edit");
    try {
      const res = await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "edit", text: editText.trim() }),
      });
      if (!res.ok) throw new Error("Edit failed");
      setDisplayText(editText.trim());
      setEditing(false);
    } catch {
      // keep editing open on failure
    } finally {
      setActing(null);
    }
  }

  const isReply = item.type === "reply" || item.type === "reply_alert";
  const accountHandle = item.x_account === "omega_memory" ? "@omega_memory" : "@jasonsosa";
  const dateStr = item.created_at || "";
  const isFutureScheduled = item.scheduled_for ? new Date(item.scheduled_for) > new Date() : false;
  const scheduledLabel = item.scheduled_for
    ? new Date(item.scheduled_for).toLocaleString("en-US", {
        month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
        timeZone: "America/New_York",
      }) + " ET"
    : null;

  return (
    <div
      className={`admin-card border-l-[3px] border-l-amber overflow-hidden transition-all duration-300 ${
        dismissed ? "opacity-0 max-h-0 mb-0 py-0 scale-95" : "max-h-[600px]"
      }`}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[14px] font-semibold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded bg-amber/15 text-amber">
            Pending tweet
          </span>
          <span className="text-[14px] text-ink-tertiary font-mono">{accountHandle}</span>
          {scheduledLabel && (
            <span className="text-[14px] text-ink-faint font-mono px-1.5 py-0.5 rounded bg-surface-elevated">
              {isFutureScheduled ? `Posts ${scheduledLabel}` : scheduledLabel}
            </span>
          )}
          {dateStr && (
            <span className="ml-auto text-[14px] text-ink-faint tabular-nums shrink-0">
              {formatTimeAgo(dateStr)}
            </span>
          )}
        </div>

        {/* Reply context */}
        {isReply && item.source_author_handle && (
          <div className="mb-2 text-[14px] text-ink-faint">
            <span>Replying to @{item.source_author_handle}</span>
            {item.source_tweet_text && (
              <p className="mt-1 pl-3 border-l-2 border-edge text-ink-faint line-clamp-2">
                {item.source_tweet_text}
              </p>
            )}
          </div>
        )}

        {/* Text or edit textarea */}
        {editing ? (
          <div className="space-y-2">
            <textarea
              ref={textareaRef}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full p-2.5 text-[15px] bg-surface border border-edge rounded-lg text-ink leading-relaxed placeholder:text-ink-faint resize-none focus:outline-none focus:border-amber/40"
              rows={4}
            />
            <div className="flex items-center gap-2 justify-end">
              <button
                onClick={() => { setEditing(false); setEditText(displayText); }}
                className="px-3 py-1.5 rounded-lg text-[14px] font-medium text-ink-tertiary hover:text-ink-secondary transition-colors min-h-[36px] touch-manipulation"
              >
                Cancel
              </button>
              <button
                onClick={handleEditSave}
                disabled={acting === "edit"}
                className="px-4 py-1.5 rounded-lg text-[14px] font-semibold bg-amber text-canvas min-h-[36px] touch-manipulation disabled:opacity-50 transition-all"
              >
                {acting === "edit" ? <Spinner /> : "Save"}
              </button>
            </div>
          </div>
        ) : (
          <p className="text-[15px] text-ink-secondary leading-relaxed whitespace-pre-wrap">
            {displayText}
          </p>
        )}

        {/* Actions */}
        {!editing && (
          <div className="flex items-center gap-2 mt-3 justify-end">
            <button
              onClick={handleReject}
              disabled={!!acting}
              className="px-4 py-2 rounded-lg text-[14px] font-medium text-ink-tertiary hover:text-ink-secondary transition-colors min-h-[44px] touch-manipulation disabled:opacity-50"
            >
              {acting === "reject" ? <Spinner /> : "Reject"}
            </button>
            <button
              onClick={() => setEditing(true)}
              disabled={!!acting}
              className="px-4 py-2 rounded-lg border border-edge text-[14px] font-medium text-ink-secondary min-h-[44px] touch-manipulation disabled:opacity-50 transition-all hover:border-edge-strong"
            >
              Edit
            </button>
            <button
              onClick={handleApprove}
              disabled={!!acting}
              className="px-4 py-2 rounded-lg bg-amber text-canvas text-[14px] font-semibold min-h-[44px] touch-manipulation disabled:opacity-50 transition-all"
            >
              {acting === "approve" ? <Spinner /> : isFutureScheduled ? "Approve" : "Approve & Post"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Proposal Card ────────────────────────────────────────────────

function hasValidEventParams(metadata: Record<string, unknown>): boolean {
  const ep = metadata.event_params;
  if (!ep || typeof ep !== "object") return false;
  const params = ep as Record<string, unknown>;
  return typeof params.event_type === "string" && params.event_type.trim().length > 0;
}

function ProposalCard({ item, onAction }: { item: ProposalItem; onAction: (id: string) => void }) {
  const [acting, setActing] = useState<"approve" | "reject" | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const canApprove = hasValidEventParams(item.metadata);

  const riskBadge = item.risk_level === "high"
    ? { bg: "bg-gold/15", text: "text-gold", label: "High risk" }
    : item.risk_level === "medium"
    ? { bg: "bg-amber/15", text: "text-amber", label: "Medium" }
    : null;

  async function handleApprove() {
    if (!canApprove) return;
    setActing("approve");
    try {
      const eventParams = (item.metadata as Record<string, unknown>).event_params;
      const res = await fetch(`/api/admin/orchestrator-feed/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve", event_params: eventParams }),
      });
      if (!res.ok) throw new Error("Approve failed");
      setDismissed(true);
      setTimeout(() => onAction(item.id), 300);
    } catch {
      setActing(null);
    }
  }

  async function handleDismiss() {
    setActing("reject");
    try {
      const res = await fetch(`/api/admin/orchestrator-feed/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reject" }),
      });
      if (!res.ok) throw new Error("Dismiss failed");
      setDismissed(true);
      setTimeout(() => onAction(item.id), 300);
    } catch {
      setActing(null);
    }
  }

  const bodyLines = item.body?.split("\n") || [];
  const isLong = bodyLines.length > 3 || (item.body?.length || 0) > 200;

  return (
    <div
      className={`admin-card border-l-[3px] border-l-type-lesson overflow-hidden transition-all duration-300 ${
        dismissed ? "opacity-0 max-h-0 mb-0 py-0 scale-95" : "max-h-[600px]"
      }`}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[14px] font-semibold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded bg-type-lesson/15 text-type-lesson">
            Proposal
          </span>
          {riskBadge && (
            <span className={`text-[14px] font-semibold px-1.5 py-0.5 rounded ${riskBadge.bg} ${riskBadge.text}`}>
              {riskBadge.label}
            </span>
          )}
          {item.source && item.source !== "admin" && (
            <span className="text-[14px] text-ink-faint">via {item.source}</span>
          )}
          <span className="ml-auto text-[14px] text-ink-faint tabular-nums shrink-0">
            {formatTimeAgo(item.created_at)}
          </span>
        </div>

        {/* Title */}
        <h4 className="text-[15px] text-ink font-medium mb-1">{item.title}</h4>

        {/* Body */}
        {item.body && (
          <>
            <div
              className={`text-[14px] text-ink-secondary leading-relaxed whitespace-pre-wrap ${
                !expanded && isLong ? "line-clamp-3" : ""
              }`}
            >
              {item.body}
            </div>
            {isLong && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-[14px] text-gold font-medium mt-1 touch-manipulation"
              >
                {expanded ? "Show less" : "Show more"}
              </button>
            )}
          </>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 mt-3 justify-end">
          {!canApprove && (
            <span className="text-[13px] text-gold/80 mr-auto">
              Missing event_params -- cannot approve
            </span>
          )}
          <button
            onClick={handleDismiss}
            disabled={!!acting}
            className="px-4 py-2 rounded-lg text-[14px] font-medium text-ink-tertiary hover:text-ink-secondary transition-colors min-h-[44px] touch-manipulation disabled:opacity-50"
          >
            {acting === "reject" ? <Spinner /> : "Dismiss"}
          </button>
          <button
            onClick={handleApprove}
            disabled={!!acting || !canApprove}
            title={!canApprove ? "This proposal is missing event_params.event_type and cannot be approved" : undefined}
            className="px-4 py-2 rounded-lg bg-gold text-canvas text-[14px] font-semibold min-h-[44px] touch-manipulation disabled:opacity-50 transition-all"
          >
            {acting === "approve" ? <Spinner /> : "Approve"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main export ──────────────────────────────────────────────────

export default function ApprovalCard({ type, item, onAction }: ApprovalCardProps) {
  if (type === "tweet") {
    return <TweetCard item={item as TweetItem} onAction={onAction} />;
  }
  return <ProposalCard item={item as ProposalItem} onAction={onAction} />;
}

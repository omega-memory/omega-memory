import { useCallback, useEffect, useState } from "react";
import { relativeTime } from "./jobUtils";

// ─── Types ─────────────────────────────────────────────

interface ApprovalItem {
  id: string;
  runId: string | null;
  scheduleLabel: string;
  approvalType: string;
  status: string;
  requestedAt: string;
  expiresAt: string | null;
  decidedAt: string | null;
  decidedBy: string | null;
  reason: string | null;
  context: Record<string, unknown> | null;
}

// ─── Helpers ────────────────────────────────────────────

function approvalTypeLabel(type: string): string {
  switch (type) {
    case "pre_execution":
      return "Pre-execution";
    case "content_review":
      return "Content review";
    case "external_action":
      return "External action";
    default:
      return type;
  }
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}

// ─── Component ──────────────────────────────────────────

export default function ApprovalsQueue() {
  const [approvals, setApprovals] = useState<ApprovalItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState<string | null>(null);
  const [actionType, setActionType] = useState<"approve" | "reject" | null>(
    null,
  );
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // ─── Fetch ──────────────────────────────────────────

  const fetchApprovals = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/approvals");
      if (!res.ok) throw new Error(`${res.status}`);
      const json = await res.json();
      setApprovals(json.approvals ?? []);
    } catch {
      setApprovals([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchApprovals();
  }, [fetchApprovals]);

  // ─── Actions ────────────────────────────────────────

  function openAction(id: string, type: "approve" | "reject") {
    setActionId(id);
    setActionType(type);
    setReason("");
    setActionError(null);
  }

  function cancelAction() {
    setActionId(null);
    setActionType(null);
    setReason("");
    setActionError(null);
  }

  async function submitAction() {
    if (!actionId || !actionType) return;
    setSubmitting(true);
    setActionError(null);

    try {
      const res = await fetch("/api/admin/approvals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: actionType,
          approvalId: actionId,
          reason: reason.trim() || undefined,
        }),
      });

      if (res.ok) {
        // Optimistically remove the card
        setApprovals((prev) => prev.filter((a) => a.id !== actionId));
        setSubmitting(false);
        cancelAction();
        return;
      }

      // Non-ok response: parse error and keep dialog open
      let errorMsg = `Action failed (${res.status})`;
      try {
        const body = await res.json();
        if (typeof body.error === "string") errorMsg = body.error;
        else if (typeof body.message === "string") errorMsg = body.message;
      } catch {
        // Response body wasn't JSON; use the status-based message
      }
      setActionError(errorMsg);
    } catch {
      setActionError("Network error. Check your connection and try again.");
    }

    setSubmitting(false);
  }

  // ─── Context Preview ───────────────────────────────

  function renderContextPreview(context: Record<string, unknown> | null) {
    if (!context) return null;

    const parts: string[] = [];

    if (typeof context.text === "string" && context.text.length > 0) {
      parts.push(truncate(context.text, 140));
    }

    if (Array.isArray(context.recipients)) {
      const count = context.recipients.length;
      parts.push(`${count} recipient${count !== 1 ? "s" : ""}`);
    }

    if (typeof context.subject === "string") {
      parts.push(`Subject: ${truncate(context.subject, 60)}`);
    }

    if (parts.length === 0) return null;

    return (
      <div className="mt-2 p-2.5 rounded-md bg-white/[0.03] border border-edge/30 text-[13px] text-ink-secondary leading-relaxed whitespace-pre-wrap break-words">
        {parts.join("\n")}
      </div>
    );
  }

  // ─── Loading ────────────────────────────────────────

  if (loading) {
    return (
      <div className="space-y-3 pt-2">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-24 rounded-lg skeleton"
            style={{ animationDelay: `${i * 120}ms` }}
          />
        ))}
      </div>
    );
  }

  // ─── Empty ──────────────────────────────────────────

  if (approvals.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-type-lesson/[0.08] flex items-center justify-center">
          <svg
            className="w-5 h-5 text-type-lesson/40"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
            />
          </svg>
        </div>
        <p className="text-[15px] text-ink-tertiary">No pending approvals</p>
      </div>
    );
  }

  // ─── Content ────────────────────────────────────────

  return (
    <div className="space-y-3 pt-2">
      {approvals.map((approval) => {
        const isActioning =
          actionId === approval.id && actionType !== null;

        return (
          <div
            key={approval.id}
            className="bg-surface-elevated rounded-lg border border-edge/50 p-4 transition-all"
          >
            {/* Header row */}
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[16px] font-medium text-ink truncate">
                    {approval.scheduleLabel}
                  </span>
                  <span className="text-[12px] font-medium px-2 py-0.5 rounded-full bg-type-reminder/[0.12] text-type-reminder">
                    {approvalTypeLabel(approval.approvalType)}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1 text-[13px] text-ink-faint">
                  <span>Requested {relativeTime(approval.requestedAt)}</span>
                  {approval.expiresAt && (
                    <span>
                      Expires {relativeTime(approval.expiresAt)}
                    </span>
                  )}
                </div>
              </div>

              {/* Action buttons (hidden when action dialog is open) */}
              {!isActioning && (
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => openAction(approval.id, "approve")}
                    className="text-[13px] font-medium px-3 py-1.5 rounded-lg bg-type-lesson/[0.12] text-type-lesson hover:bg-type-lesson/[0.2] transition-colors min-h-[36px] touch-manipulation"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => openAction(approval.id, "reject")}
                    className="text-[13px] font-medium px-3 py-1.5 rounded-lg bg-type-error/[0.1] text-type-error hover:bg-type-error/[0.18] transition-colors min-h-[36px] touch-manipulation"
                  >
                    Reject
                  </button>
                </div>
              )}
            </div>

            {/* Context preview */}
            {renderContextPreview(approval.context)}

            {/* Action dialog (inline) */}
            {isActioning && (
              <div className="mt-3 pt-3 border-t border-edge/30">
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`text-[13px] font-medium ${
                      actionType === "approve"
                        ? "text-type-lesson"
                        : "text-type-error"
                    }`}
                  >
                    {actionType === "approve"
                      ? "Approve this job?"
                      : "Reject this job?"}
                  </span>
                </div>
                <input
                  type="text"
                  placeholder="Reason (optional)"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") submitAction();
                    if (e.key === "Escape") cancelAction();
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-edge/40 text-[14px] text-ink placeholder:text-ink-faint/50 focus:outline-none focus:border-ink-tertiary/40 transition-colors"
                  autoFocus
                />
                {actionError && (
                  <div className="mt-2 px-3 py-2 rounded-lg bg-type-error/[0.08] border border-type-error/20 text-[13px] text-type-error">
                    {actionError}
                  </div>
                )}
                <div className="flex items-center justify-end gap-2 mt-2">
                  <button
                    onClick={cancelAction}
                    disabled={submitting}
                    className="text-[13px] px-3 py-1.5 rounded-lg text-ink-tertiary hover:text-ink transition-colors min-h-[36px] touch-manipulation"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={submitAction}
                    disabled={submitting}
                    className={`text-[13px] font-medium px-4 py-1.5 rounded-lg min-h-[36px] touch-manipulation transition-colors ${
                      actionType === "approve"
                        ? "bg-type-lesson/[0.15] text-type-lesson hover:bg-type-lesson/[0.25]"
                        : "bg-type-error/[0.12] text-type-error hover:bg-type-error/[0.2]"
                    } ${submitting ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {submitting
                      ? "..."
                      : actionType === "approve"
                        ? "Confirm Approve"
                        : "Confirm Reject"}
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

import { useCallback } from "react";
import {
  parseContentRich,
  extractContentStats,
  humanProject,
  timeAgo,
  ExpandedContent,
  GrantStatsBar,
  isConfidentialContent,
} from "../contentUtils";
import type { Memory } from "./feedScoring";
import {
  getTypeVisual,
  TypeIcon,
  formatDue,
  RELEVANCE_CHECK_TYPES,
  RELEVANCE_OPTIONS,
} from "./feedScoring";

interface FeedDetailProps {
  m: Memory;
  isOpen: boolean;
  onClose: () => void;
  onNavigateToApprovals?: () => void;
  reminderActing: Set<string>;
  onDismiss: (m: Memory) => void;
  onExtend: (m: Memory) => void;
  onInteract: (m: Memory, payload: Record<string, any>) => void;
  onArchive: (m: Memory) => void;
  onUnarchive: (m: Memory) => void;
  confidentialUnlocked: boolean;
  onUnlockRequest: () => void;
}

export default function FeedDetail({
  m,
  isOpen,
  onClose,
  onNavigateToApprovals,
  reminderActing,
  onDismiss,
  onExtend,
  onInteract,
  onArchive,
  onUnarchive,
  confidentialUnlocked,
  onUnlockRequest,
}: FeedDetailProps) {
  const parsed = parseContentRich(m.content);
  const isConfidential = isConfidentialContent(m.content, m.metadata);
  const isGrant = parsed.category === "grant";
  const isXBrief = m.content.toLowerCase().startsWith("x brief") || m.metadata?.tag === "x_brief";
  const projectName = humanProject(m.project);
  const typeVisual = getTypeVisual(m.event_type, m.metadata?.card_type, isXBrief, isGrant);
  const isReminder = m.event_type === "reminder";
  const isDismissed = m.metadata?.reminder_status === "dismissed";
  const isTask = m.event_type === "task_completion";
  const isActing = reminderActing.has(m.id);
  const dueDate = parsed.dueDate || (m.metadata?.remind_at ? new Date(m.metadata.remind_at) : null);
  const isOverdue = dueDate ? (dueDate.getTime() - Date.now()) < 0 : false;

  const isRatable = m.metadata?.ratable && !m.metadata?.rating;

  const ageHours = (Date.now() - new Date(m.created_at).getTime()) / 3_600_000;
  const needsRelevanceCheck =
    RELEVANCE_CHECK_TYPES.includes(m.event_type || "") &&
    ageHours >= 12 &&
    !m.metadata?.relevance_status;

  const showConfidentialOverlay = isConfidential && !confidentialUnlocked;

  return (
    <div
      className="transition-[grid-template-rows] duration-300 ease-out overflow-hidden"
      style={{ display: "grid", gridTemplateRows: isOpen ? "1fr" : "0fr" }}
    >
      <div className="min-h-0 overflow-hidden">
        <div className="rounded-xl border border-edge bg-surface mx-3 mb-2 shadow-card overflow-hidden">
          {/* Header */}
          <div className="px-5 pt-4 pb-3">
            <div className="flex items-center gap-2 mb-3">
              <span className={`${typeVisual.colorClass} opacity-80`}>
                <TypeIcon d={typeVisual.iconPath} />
              </span>
              <span className={`text-[14px] font-semibold font-mono uppercase tracking-[0.10em] ${typeVisual.colorClass} opacity-70`}>
                {typeVisual.label}
              </span>
              {projectName && (
                <>
                  <span className="text-[14px] text-ink-faint">&middot;</span>
                  <span className="text-[14px] text-ink-faint">{projectName}</span>
                </>
              )}
              {m.metadata?.source_url && (
                <>
                  <span className="text-[14px] text-ink-faint">&middot;</span>
                  <a
                    href={m.metadata.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[14px] text-ink-faint hover:text-ink-secondary transition-colors truncate max-w-[120px] flex items-center gap-0.5"
                  >
                    {m.metadata.source_domain || new URL(m.metadata.source_url).hostname.replace(/^www\./, "")}
                    <svg className="w-2.5 h-2.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                    </svg>
                  </a>
                </>
              )}
              <div className="ml-auto flex items-center gap-2">
                {dueDate && !showConfidentialOverlay && (
                  <span className={`text-[14px] font-semibold px-1.5 py-0.5 rounded-full ${
                    isOverdue
                      ? "bg-type-error/15 text-type-error"
                      : "bg-type-reminder/15 text-type-reminder"
                  }`}>
                    {formatDue(dueDate)}
                  </span>
                )}
                <span className="text-[14px] text-ink-faint font-mono">{timeAgo(m.created_at)}</span>
                <button
                  onClick={onClose}
                  className="p-1 rounded-lg text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated transition-all touch-manipulation"
                  aria-label="Close"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Full title */}
            <p className="text-[18px] font-medium text-ink leading-snug mb-2">
              {parsed.title}
            </p>

            {/* Content */}
            {showConfidentialOverlay ? (
              <div className="relative">
                <div className="confidential-blur" aria-hidden="true">
                  <p className="text-[15px] text-ink-secondary leading-relaxed">
                    This content contains sensitive information that requires authentication to view.
                  </p>
                </div>
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 py-3">
                  <svg className="w-5 h-5 text-gold/60" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
                  </svg>
                  <button
                    onClick={onUnlockRequest}
                    className="text-[14px] font-semibold px-3.5 py-1.5 rounded-lg bg-gold/[0.08] border border-gold/20 text-gold hover:bg-gold/[0.14] transition-colors touch-manipulation"
                  >
                    Touch ID to view
                  </button>
                </div>
              </div>
            ) : (
              <div className={isConfidential ? "confidential-revealed" : ""}>
                <ExpandedContent parsed={parsed} raw={m.content} eventType={m.event_type} />
              </div>
            )}

            {/* Grant stats */}
            {!showConfidentialOverlay && isGrant && (
              <div className="mt-2">
                <GrantStatsBar content={m.content} />
              </div>
            )}

            {/* Relevance check */}
            {!showConfidentialOverlay && needsRelevanceCheck && (
              <div className="mt-3 pt-2 border-t border-edge/40">
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-[14px] text-ink-tertiary">
                    {RELEVANCE_OPTIONS[m.event_type || ""]?.label}
                  </p>
                  <button
                    className="text-[14px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation"
                    disabled={isActing}
                    onClick={() => onInteract(m, {
                      relevance_status: "skipped",
                      relevance_checked_at: new Date().toISOString(),
                    })}
                  >
                    Skip
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(RELEVANCE_OPTIONS[m.event_type || ""]?.options || []).map((opt) => (
                    <button
                      key={opt}
                      className="pill-option"
                      disabled={isActing}
                      onClick={() => onInteract(m, {
                        relevance_status: opt,
                        relevance_checked_at: new Date().toISOString(),
                      })}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {!showConfidentialOverlay && m.metadata?.relevance_status && (
              <div className="mt-2">
                <p className="text-[14px] text-gold flex items-center gap-1.5">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                  {m.metadata.relevance_status}
                </p>
              </div>
            )}

            {/* Action row */}
            <div className="flex items-center gap-2 mt-3 pt-2 border-t border-edge/40 text-[14px]">
              {/* Archive / Unarchive */}
              {!showConfidentialOverlay && (
                m.metadata?.archived_at ? (
                  <button
                    onClick={() => onUnarchive(m)}
                    className="text-[14px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation flex items-center gap-1 py-0.5"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0-3-3m3 3 3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z" />
                    </svg>
                    Unarchive
                  </button>
                ) : (
                  <button
                    onClick={() => onArchive(m)}
                    className="text-[14px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation flex items-center gap-1 py-0.5"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z" />
                    </svg>
                    Archive
                  </button>
                )
              )}

              {/* Rating thumbs */}
              {!showConfidentialOverlay && isRatable && (
                <span className="flex items-center gap-0.5">
                  <button
                    className="rating-thumb"
                    disabled={isActing}
                    onClick={() => onInteract(m, { rating: "up", rated_at: new Date().toISOString() })}
                    aria-label="Thumbs up"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V3a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m7.594-9.75H15" />
                    </svg>
                  </button>
                  <button
                    className="rating-thumb"
                    disabled={isActing}
                    onClick={() => onInteract(m, { rating: "down", rated_at: new Date().toISOString() })}
                    aria-label="Thumbs down"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715A12.137 12.137 0 0 1 2.25 12c0-2.848.992-5.464 2.649-7.521C5.287 3.997 5.886 3.75 6.504 3.75h4.016a4.5 4.5 0 0 1 1.423.23l3.114 1.04a4.5 4.5 0 0 0 1.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 0 0 7.5 19.75 2.25 2.25 0 0 0 9.75 22a.75.75 0 0 0 .75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 0 0 2.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384" />
                    </svg>
                  </button>
                </span>
              )}
              {!showConfidentialOverlay && m.metadata?.rating && (
                <span className="text-[14px] text-gold">
                  {m.metadata.rating === "up" ? "+" : "-"}
                </span>
              )}

              {!showConfidentialOverlay && isTask && onNavigateToApprovals && (
                <button
                  onClick={onNavigateToApprovals}
                  className="text-[14px] font-medium text-type-task hover:text-type-task/80 transition-colors touch-manipulation py-0.5"
                >
                  Review
                </button>
              )}

              <div className="ml-auto flex items-center gap-1.5">
                {!showConfidentialOverlay && isReminder && !isDismissed && (
                  <>
                    <button
                      onClick={() => onExtend(m)}
                      disabled={isActing}
                      className="text-[14px] font-medium px-3.5 py-1 rounded-full bg-type-reminder/10 text-type-reminder hover:bg-type-reminder/20 transition-colors touch-manipulation disabled:opacity-50"
                    >
                      Snooze
                    </button>
                    <button
                      onClick={() => onDismiss(m)}
                      disabled={isActing}
                      className="text-[14px] font-medium px-3.5 py-1 rounded-full bg-type-lesson/10 text-type-lesson hover:bg-type-lesson/20 transition-colors touch-manipulation disabled:opacity-50"
                    >
                      Done
                    </button>
                  </>
                )}
                {!showConfidentialOverlay && isReminder && isDismissed && (
                  <span className="text-[14px] text-ink-faint font-medium px-2 py-0.5 rounded-full bg-surface-elevated">
                    Dismissed
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

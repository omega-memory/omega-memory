import { parseContentRich, humanProject, timeAgo, humanizeBullet } from "../contentUtils";
import type { Memory, TypeVisual, ExpansionMode } from "./feedScoring";
import { getTypeVisual, TypeIcon, formatDue } from "./feedScoring";

interface FeedRowProps {
  m: Memory;
  isExpanded: boolean;
  onToggleExpand: (id: string) => void;
  isNeedsAttention?: boolean;
  isRead?: boolean;
  expansionMode?: ExpansionMode;
  density?: "default" | "compact";
  isFocused?: boolean;
  hideTimestamp?: boolean;
}

export default function FeedRow({ m, isExpanded, onToggleExpand, isNeedsAttention, isRead, expansionMode = "full", density = "default", isFocused, hideTimestamp }: FeedRowProps) {
  const parsed = parseContentRich(m.content);
  const isGrant = parsed.category === "grant";
  const isXBrief = m.content.toLowerCase().startsWith("x brief") || m.metadata?.tag === "x_brief";
  const projectName = humanProject(m.project);
  const typeVisual = getTypeVisual(m.event_type, m.metadata?.card_type, isXBrief, isGrant);

  const isSessionRecap = !!m.metadata?.is_session_recap;
  const hasUnreadNextSteps = isSessionRecap && m.metadata?.session_achievement?.has_unread_next_steps;

  const dueDate = parsed.dueDate || (m.metadata?.remind_at ? new Date(m.metadata.remind_at) : null);
  const isOverdue = dueDate ? (dueDate.getTime() - Date.now()) < 0 : false;
  const isDismissed = m.metadata?.reminder_status === "dismissed";

  const ageHours = (Date.now() - new Date(m.created_at).getTime()) / 3_600_000;
  const needsRelevanceCheck =
    !isSessionRecap &&
    ["decision", "lesson_learned", "error_pattern"].includes(m.event_type || "") &&
    ageHours >= 12 &&
    !m.metadata?.relevance_status;

  const showActionDot = hasUnreadNextSteps || needsRelevanceCheck;
  const isExpandable = expansionMode !== "none";

  // W1: Apply background tint only to high-signal types that need to visually pop
  const rowTintMap: Record<string, string> = {
    error_pattern: "row-tint-error",
    reminder: "row-tint-reminder",
  };
  const getRowTint = (): string => {
    if (isRead || isDismissed) return "";
    if (hasUnreadNextSteps) return "row-tint-interactive";
    if (isXBrief) return "row-tint-xbrief";
    if (isGrant) return "row-tint-grant";
    return rowTintMap[m.event_type || ""] || "";
  };
  const rowTint = getRowTint();

  const isCompact = density === "compact";
  const titleSize = isCompact ? "text-[16px]" : "text-[18px]";
  const metaSize = isCompact ? "text-[15px]" : "text-[15px]";
  const iconSize = isCompact ? "w-3.5 h-3.5" : "w-4 h-4";

  const rowContent = (
    <>
      {/* Type badge pill */}
      <span
        className={`shrink-0 inline-flex items-center gap-1 rounded-full ${isCompact ? "px-1.5 py-0.5" : "px-2 py-0.5"} text-[12px] font-semibold uppercase tracking-wider ${
          showActionDot
            ? "bg-gold/15 text-gold shadow-[0_0_6px_rgba(212,168,67,0.3)] animate-pulse"
            : isOverdue
            ? "bg-type-error/10 text-type-error"
            : isRead
            ? "bg-ink-faint/8 text-ink-faint"
            : `${typeVisual.colorClass} bg-current/[0.08]`
        }`}
        style={!showActionDot && !isOverdue && !isRead ? { backgroundColor: `color-mix(in srgb, currentColor 8%, transparent)` } : undefined}
      >
        <TypeIcon d={typeVisual.iconPath} className={isCompact ? "w-3 h-3" : "w-3.5 h-3.5"} />
        {!isCompact && <span>{typeVisual.label}</span>}
      </span>

      {/* Title + inline detail */}
      <span className="flex-1 min-w-0">
        <span className={`${titleSize} truncate block leading-snug ${isRead ? "text-ink-secondary" : "text-ink"}`}>
          {parsed.title}
        </span>
        {!isExpanded && parsed.detail && !isCompact && (
          <span className={`${metaSize} text-ink-tertiary truncate block leading-snug mt-0.5 ${
            isRead ? "hidden sm:block" : ""
          }`}>
            {humanizeBullet(parsed.detail).slice(0, 80)}
          </span>
        )}
      </span>

      {/* Priority badge for high-priority items (4-5) */}
      {m.priority >= 4 && !isRead && (
        <span className={`text-[12px] font-semibold px-2 py-0.5 rounded-full shrink-0 ${
          m.priority >= 5
            ? "bg-type-error/10 text-type-error"
            : "bg-gold/10 text-gold"
        }`}>
          {m.priority >= 5 ? "Urgent" : "High"}
        </span>
      )}

      {/* Due date badge */}
      {dueDate && (
        <span className={`text-[14px] font-semibold px-1.5 py-0.5 rounded-full shrink-0 ${
          isOverdue
            ? "bg-type-error/15 text-type-error"
            : "bg-type-reminder/15 text-type-reminder"
        }`}>
          {formatDue(dueDate)}
        </span>
      )}

      {/* Project pill */}
      {projectName && (
        <span className="text-[12px] text-ink-tertiary shrink-0 hidden sm:inline-flex items-center px-2 py-0.5 rounded-full border border-edge font-medium truncate max-w-[140px]">
          {projectName}
        </span>
      )}

      {/* Time */}
      {!hideTimestamp && (
        <span className={`${metaSize} text-ink-faint font-mono shrink-0`}>
          {timeAgo(m.created_at)}
        </span>
      )}
    </>
  );

  const baseClasses = `w-full flex items-start gap-2.5 ${isCompact ? "px-2.5 py-1.5" : "px-3 py-2.5"} text-left transition-colors rounded-lg group
    ${isExpanded ? "bg-surface-elevated" : isExpandable ? "hover:bg-surface-hover" : ""}
    ${isNeedsAttention ? "border-l-[3px] border-l-gold/60" : ""}
    ${isDismissed ? "opacity-45" : ""}
    ${isFocused ? "ring-1 ring-gold/40 bg-gold/[0.03]" : ""}
    ${rowTint}`;

  if (!isExpandable) {
    return <div className={baseClasses}>{rowContent}</div>;
  }

  return (
    <button onClick={() => onToggleExpand(m.id)} aria-label={`${typeVisual.label}: ${parsed.title}`} className={baseClasses}>
      {rowContent}
    </button>
  );
}

import type { Memory } from "./feedScoring";
import { parseContentRich, humanizeBullet } from "../contentUtils";

interface FeedInlineDetailProps {
  m: Memory;
  isOpen: boolean;
  onArchive?: (m: Memory) => void;
  onDismiss?: (m: Memory) => void;
}

export default function FeedInlineDetail({ m, isOpen, onArchive, onDismiss }: FeedInlineDetailProps) {
  const parsed = parseContentRich(m.content);
  const rawText = parsed.detail || m.content;
  const text = humanizeBullet(rawText);
  const isReminder = m.event_type === "reminder";
  const isDismissed = m.metadata?.reminder_status === "dismissed";
  const isArchived = !!m.metadata?.archived_at;

  return (
    <div
      className="transition-[grid-template-rows] duration-300 ease-out overflow-hidden"
      style={{ display: "grid", gridTemplateRows: isOpen ? "1fr" : "0fr" }}
    >
      <div className="min-h-0 overflow-hidden">
        <div className="px-10 pb-2">
          <p className="text-[15px] text-ink-secondary leading-relaxed line-clamp-3">
            {text}
          </p>
          {/* Inline action buttons */}
          <div className="flex items-center gap-3 mt-1.5 text-[13px]">
            {onArchive && !isArchived && (
              <button
                onClick={() => onArchive(m)}
                className="text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation flex items-center gap-1"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z" />
                </svg>
                Archive
              </button>
            )}
            {onDismiss && isReminder && !isDismissed && (
              <button
                onClick={() => onDismiss(m)}
                className="text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation flex items-center gap-1"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
                Dismiss
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

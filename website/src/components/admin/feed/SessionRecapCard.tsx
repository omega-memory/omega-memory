import { useState } from "react";
import { humanProject, timeAgo, humanizeBullet } from "../contentUtils";
import { TypeIcon, TYPE_VISUALS } from "./feedScoring";

interface SessionAchievement {
  session_id: string;
  project: string | null;
  task: string | null;
  started_at: string;
  duration_seconds: number | null;
  completed_tasks: string[];
  next_steps: string[];
  files_modified: string[];
  decisions_made: string[];
  blocked_items: string[];
  git_commits: { hash: string; message: string }[];
  has_unread_next_steps: boolean;
}

interface SessionRecapCardProps {
  achievement: SessionAchievement;
  isOpen: boolean;
  onClose: () => void;
  onMarkNextStepRead: (sessionId: string) => void;
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return "";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

export default function SessionRecapCard({ achievement, isOpen, onClose, onMarkNextStepRead }: SessionRecapCardProps) {
  const [showCommits, setShowCommits] = useState(false);
  const a = achievement;
  const visual = TYPE_VISUALS.session_recap;
  const projectName = humanProject(a.project);

  return (
    <div
      className="transition-[grid-template-rows] duration-300 ease-out overflow-hidden"
      style={{ display: "grid", gridTemplateRows: isOpen ? "1fr" : "0fr" }}
    >
      <div className="min-h-0 overflow-hidden">
        <div className="rounded-xl border border-edge bg-surface mx-3 mb-2 shadow-card overflow-hidden">
          <div className="px-5 pt-4 pb-3">
            {/* Header */}
            <div className="flex items-center gap-2 mb-3">
              <span className={`${visual.colorClass} opacity-80`}>
                <TypeIcon d={visual.iconPath} />
              </span>
              <span className={`text-[14px] font-semibold font-mono uppercase tracking-[0.10em] ${visual.colorClass} opacity-70`}>
                Session
              </span>
              {projectName && (
                <>
                  <span className="text-[14px] text-ink-faint">&middot;</span>
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full border border-edge text-[11px] font-medium text-ink-tertiary">
                    {projectName}
                  </span>
                </>
              )}
              {a.duration_seconds && (
                <>
                  <span className="text-[14px] text-ink-faint">&middot;</span>
                  <span className="text-[14px] text-ink-faint font-mono">{formatDuration(a.duration_seconds)}</span>
                </>
              )}
              <div className="ml-auto flex items-center gap-2">
                <span className="text-[14px] text-ink-faint font-mono">{timeAgo(a.started_at)}</span>
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

            {/* Task title */}
            {a.task && (
              <p className="text-[18px] font-medium text-ink leading-snug mb-3">
                {humanizeBullet(a.task.replace(/https?:\/\/[^\s]+/g, (url) => {
                  try { const u = new URL(url); return u.searchParams.get("tab") || u.pathname.split("/").pop() || ""; }
                  catch { return ""; }
                }))}
              </p>
            )}

            {/* Completed tasks */}
            {a.completed_tasks.length > 0 && (
              <div className="mb-3">
                <ul className="space-y-1">
                  {a.completed_tasks.map((task, i) => (
                    <li key={i} className="flex items-start gap-2 text-[15px] text-ink-secondary leading-snug">
                      <svg className="w-4 h-4 text-type-task shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                      </svg>
                      {humanizeBullet(task)}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Stats row */}
            {(a.files_modified.length > 0 || a.decisions_made.length > 0 || a.git_commits.length > 0) && (
              <div className="flex flex-wrap gap-1.5 mb-3">
                {a.files_modified.length > 0 && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-elevated border border-edge text-[12px] font-medium text-ink-tertiary">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                    </svg>
                    {a.files_modified.length} {a.files_modified.length === 1 ? "file changed" : "files changed"}
                  </span>
                )}
                {a.decisions_made.length > 0 && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-elevated border border-edge text-[12px] font-medium text-ink-tertiary">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z" />
                    </svg>
                    {a.decisions_made.length} {a.decisions_made.length === 1 ? "decision" : "decisions"}
                  </span>
                )}
                {a.git_commits.length > 0 && (
                  <button
                    onClick={() => setShowCommits(!showCommits)}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-elevated border border-edge text-[12px] font-medium text-ink-tertiary hover:text-ink-secondary transition-colors"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75 22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3-4.5 16.5" />
                    </svg>
                    {a.git_commits.length} commits
                    <svg className={`w-2.5 h-2.5 transition-transform ${showCommits ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                    </svg>
                  </button>
                )}
              </div>
            )}

            {/* Git commits (collapsible) */}
            {showCommits && a.git_commits.length > 0 && (
              <div className="mb-3 pl-2 border-l-2 border-edge">
                {a.git_commits.map((c, i) => (
                  <div key={i} className="flex items-start gap-2 py-1 text-[13px]">
                    <code className="text-ink-faint font-mono shrink-0">{c.hash.slice(0, 7)}</code>
                    <span className="text-ink-secondary">{c.message}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Next steps */}
            {Array.isArray(a.next_steps) && a.next_steps.length > 0 && (
              <div className={`rounded-lg p-3 mb-3 ${a.has_unread_next_steps ? "bg-gold/[0.06] border border-gold/15" : "bg-surface-elevated border border-edge"}`}>
                <div className="flex items-center gap-1.5 mb-2">
                  <span className={`text-[12px] font-semibold uppercase tracking-wider ${a.has_unread_next_steps ? "text-gold" : "text-ink-tertiary"}`}>
                    Next Steps
                  </span>
                  {a.has_unread_next_steps && (
                    <span className="w-1.5 h-1.5 rounded-full bg-gold shadow-[0_0_4px_rgba(212,168,67,0.5)]" />
                  )}
                </div>
                <ul className="space-y-1">
                  {(a.next_steps || []).map((step, i) => (
                    <li key={i} className={`text-[14px] leading-snug ${a.has_unread_next_steps ? "text-ink" : "text-ink-secondary"}`}>
                      <span className="text-ink-faint mr-1.5">-</span>
                      {step}
                    </li>
                  ))}
                </ul>
                {a.has_unread_next_steps && (
                  <button
                    onClick={() => onMarkNextStepRead(a.session_id)}
                    className="mt-2 text-[13px] font-medium text-gold hover:text-gold/80 transition-colors touch-manipulation"
                  >
                    Mark reviewed
                  </button>
                )}
              </div>
            )}

            {/* Blocked items */}
            {a.blocked_items.length > 0 && (
              <div className="rounded-lg p-3 bg-type-error/[0.04] border border-type-error/10">
                <span className="text-[12px] font-semibold uppercase tracking-wider text-type-error mb-2 block">
                  Blocked
                </span>
                <ul className="space-y-1">
                  {a.blocked_items.map((item, i) => (
                    <li key={i} className="text-[14px] text-type-error/80 leading-snug">
                      <span className="mr-1.5">-</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Decisions */}
            {a.decisions_made.length > 0 && (
              <div className="mt-3">
                <span className="text-[12px] font-semibold uppercase tracking-wider text-ink-tertiary mb-1.5 block">Decisions</span>
                <ul className="space-y-1">
                  {a.decisions_made.map((d, i) => (
                    <li key={i} className="text-[14px] text-ink-secondary leading-snug">
                      <span className="text-ink-faint mr-1.5">-</span>
                      {humanizeBullet(d)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

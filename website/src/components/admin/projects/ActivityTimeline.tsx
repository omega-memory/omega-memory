import { useState } from "react";
import type { DecisionItem, ActivityItem } from "./types";
import { relativeTime, humanize, humanizeDecision, isNoiseActivity } from "./utils";

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg className={`w-4 h-4 text-ink-tertiary transition-transform duration-150 ${open ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
    </svg>
  );
}

interface ActivityTimelineProps {
  activity: ActivityItem[];
  decisions: DecisionItem[];
}

export default function ActivityTimeline({ activity, decisions }: ActivityTimelineProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  // Filter noise and humanize
  const cleanActivity = activity
    .filter((a) => !isNoiseActivity(a.summary) && !isNoiseActivity(humanize(a.summary)))
    .map((a) => ({ ...a, clean: humanize(a.summary) }))
    .filter((a) => a.clean.length >= 10 && a.clean !== "Session");

  const typeLabel = (type: string) => {
    if (type === "commit") return "Commit";
    if (type === "decision") return "Decision";
    return "Session";
  };
  const typeBadgeCls = (type: string) => {
    if (type === "commit") return "bg-type-lesson/10 text-type-lesson";
    if (type === "decision") return "bg-type-reminder/10 text-type-reminder";
    return "bg-ink-faint/10 text-ink-faint";
  };

  return (
    <div>
      {cleanActivity.length > 0 ? (
        <>
          <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
            Recent Work
          </span>
          <div className="mt-2 space-y-0.5">
            {cleanActivity.slice(0, 8).map((a, i) => {
              const isOpen = expandedIdx === i;
              const isFeatureBranch = a.branch && !["main", "master", "develop"].includes(a.branch);
              const hasDetail = !!(a.detail || (a.filesChanged && a.filesChanged.length > 0) || isFeatureBranch);
              return (
                <div key={i}>
                  <button
                    onClick={() => hasDetail ? setExpandedIdx(isOpen ? null : i) : undefined}
                    className={`w-full text-left flex items-start gap-2.5 py-2 px-2 rounded-lg transition-colors ${
                      hasDetail ? "cursor-pointer hover:bg-surface-elevated/30" : "cursor-default"
                    } ${isOpen ? "bg-surface-elevated/30" : ""}`}
                  >
                    <span className={`text-[12px] px-1.5 py-0.5 rounded font-mono shrink-0 mt-0.5 ${typeBadgeCls(a.type)}`}>
                      {typeLabel(a.type)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <span className="text-[15px] text-ink-secondary leading-snug">
                        {a.clean.length > 80 ? a.clean.slice(0, 77) + "..." : a.clean}
                      </span>
                      <span className="text-[12px] text-ink-faint font-mono ml-2">{relativeTime(a.timestamp)}</span>
                    </div>
                    {hasDetail && (
                      <ChevronIcon open={isOpen} />
                    )}
                  </button>
                  {isOpen && hasDetail && (
                    <div className="ml-[72px] pb-3 space-y-2">
                      {a.detail && (
                        <p className="text-[14px] text-ink-secondary/80 leading-relaxed">
                          {humanize(a.detail)}
                        </p>
                      )}
                      {isFeatureBranch && (
                        <div className="flex items-center gap-1.5">
                          <span className="text-[12px] text-ink-faint font-mono">branch:</span>
                          <span className="text-[13px] text-ink-secondary font-mono bg-surface-elevated px-1.5 py-0.5 rounded">
                            {a.branch}
                          </span>
                        </div>
                      )}
                      {a.filesChanged && a.filesChanged.length > 0 && (
                        <div>
                          <span className="text-[12px] text-ink-faint font-mono">
                            {a.filesChanged.length} file{a.filesChanged.length > 1 ? "s" : ""} changed
                          </span>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {a.filesChanged.slice(0, 6).map((f, fi) => {
                              const fileName = f.split("/").pop() ?? f;
                              return (
                                <span key={fi} className="text-[12px] font-mono text-ink-faint bg-surface-elevated px-1.5 py-0.5 rounded">
                                  {fileName}
                                </span>
                              );
                            })}
                            {a.filesChanged.length > 6 && (
                              <span className="text-[12px] text-ink-faint">+{a.filesChanged.length - 6} more</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <span className="text-[15px] text-ink-faint">No recent activity</span>
      )}
      {decisions.length > 0 && (
        <div className="mt-4">
          <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
            Key Decisions
          </span>
          <div className="mt-2 space-y-2">
            {decisions.map((d: DecisionItem) => (
              <div key={d.id} className="border-b border-edge-subtle/30 pb-2 last:border-0">
                <span className="text-[12px] text-ink-faint font-mono">{relativeTime(d.createdAt)}</span>
                <p className="text-[15px] text-ink-secondary leading-snug mt-0.5">{humanizeDecision(d.decision)}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

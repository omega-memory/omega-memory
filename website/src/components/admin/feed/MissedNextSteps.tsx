import { humanProject } from "../contentUtils";

interface SessionAchievement {
  session_id: string;
  project: string | null;
  next_steps: string[];
  has_unread_next_steps: boolean;
}

interface MissedNextStepsProps {
  achievements: SessionAchievement[];
  onMarkRead: (sessionId: string) => void;
}

export default function MissedNextSteps({ achievements, onMarkRead }: MissedNextStepsProps) {
  // Only show achievements with unread next steps
  const unread = achievements.filter((a) => a.has_unread_next_steps && a.next_steps.length > 0);
  if (unread.length === 0) return null;

  // Group by project
  const byProject = new Map<string, typeof unread>();
  for (const a of unread) {
    const key = a.project || "unknown";
    const list = byProject.get(key) || [];
    list.push(a);
    byProject.set(key, list);
  }

  return (
    <div className="mb-2">
      <div className="pt-3 pb-2 flex items-center gap-2 sticky top-0 z-[5] bg-canvas/80 backdrop-blur-md">
        <span className="w-2 h-2 rounded-full bg-gold shadow-[0_0_8px_rgba(212,168,67,0.5)]" />
        <h3 className="text-[14px] font-semibold text-gold uppercase tracking-[0.1em]">
          Missed Next Steps
        </h3>
        <span className="text-[14px] text-gold/60 font-medium tabular-nums">
          {unread.reduce((sum, a) => sum + a.next_steps.length, 0)}
        </span>
        <div className="flex-1 h-px bg-gradient-to-r from-gold/10 to-transparent ml-1" />
      </div>

      <div className="space-y-2">
        {Array.from(byProject.entries()).map(([project, sessions]) => {
          const projectName = humanProject(project);
          return (
            <div key={project} className="rounded-lg bg-gold/[0.03] border border-gold/10 p-3">
              {projectName && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full border border-edge text-[11px] font-medium text-ink-tertiary mb-2">
                  {projectName}
                </span>
              )}
              <ul className="space-y-1.5">
                {sessions.flatMap((session) =>
                  (Array.isArray(session.next_steps) ? session.next_steps : []).map((step, i) => (
                    <li
                      key={`${session.session_id}-${i}`}
                      className="flex items-start gap-2 text-[14px] text-ink leading-snug"
                    >
                      <span className="text-gold/60 mt-0.5 shrink-0">-</span>
                      <span className="flex-1">{step}</span>
                    </li>
                  ))
                )}
              </ul>
              {sessions.map((session) => (
                <button
                  key={session.session_id}
                  onClick={() => onMarkRead(session.session_id)}
                  className="mt-2 mr-2 text-[13px] font-medium text-gold hover:text-gold/80 transition-colors touch-manipulation"
                >
                  Mark reviewed
                </button>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

import type { ProjectOverview } from "./types";
import { relativeTime, HEALTH_THEME, humanize } from "./utils";

interface DetailHeaderProps {
  project: ProjectOverview;
}

export default function DetailHeader({ project }: DetailHeaderProps) {
  const theme = HEALTH_THEME[project.health];

  const totalTasks =
    project.tasks.pending +
    project.tasks.inProgress +
    project.tasks.completed;
  const completedPct = totalTasks > 0
    ? Math.round((project.tasks.completed / totalTasks) * 100)
    : 0;

  return (
    <div className="space-y-3 pb-4 border-b border-white/[0.06]">
      {/* Health + Name */}
      <div className="flex items-center gap-2">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${theme.badge}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${theme.dot}`} />
          {theme.label}
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/[0.06] text-white/30">
          {project.category}
        </span>
        <span className="ml-auto text-xs text-white/25">
          {relativeTime(project.lastActive)}
        </span>
      </div>

      <h2 className="text-lg font-semibold text-white/90">{project.name}</h2>

      {/* AI Summary */}
      {project.latestHandoff?.keyContext && (
        <p className="text-sm text-white/50 leading-relaxed">
          {humanize(
            project.latestHandoff.keyContext.length > 200
              ? project.latestHandoff.keyContext.slice(0, 200) + "..."
              : project.latestHandoff.keyContext,
          )}
        </p>
      )}

      {/* Git branch */}
      {project.gitBranch && (
        <div className="flex items-center gap-1.5 text-xs text-white/30">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M3 2v4m0 0a2 2 0 104 0M3 6a2 2 0 014 0m0 0V2m0 8v-2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
          <code className="font-mono text-white/40">{project.gitBranch}</code>
        </div>
      )}

      {/* Task progress bar */}
      {totalTasks > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] text-white/30">
            <span>Tasks</span>
            <span>{project.tasks.completed}/{totalTasks} ({completedPct}%)</span>
          </div>
          <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
            <div
              className="h-full rounded-full bg-green-400/60 transition-all"
              style={{ width: `${completedPct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

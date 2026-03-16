import type { ProjectOverview } from "./types";
import { relativeTime, HEALTH_THEME, getContextLine, computeMomentum, MOMENTUM_THEME } from "./utils";
import Sparkline from "./Sparkline";

interface ProjectOverviewCardProps {
  project: ProjectOverview;
  onClick: () => void;
}

const CONTEXT_STYLE = {
  blocker: "text-red-400/70",
  next: "text-amber-400/70",
  summary: "text-white/50",
} as const;

export default function ProjectOverviewCard({
  project,
  onClick,
}: ProjectOverviewCardProps) {
  const theme = HEALTH_THEME[project.health];
  const context = getContextLine(project);
  const momentum = computeMomentum(project);
  const mTheme = MOMENTUM_THEME[momentum];
  const sparkColor = momentum === "up" ? "rgba(74, 222, 128, 0.6)"
    : momentum === "cooling" ? "rgba(251, 191, 36, 0.6)"
    : momentum === "stalled" ? "rgba(248, 113, 113, 0.5)"
    : "rgba(255, 255, 255, 0.3)";

  const blockedCount = project.tasks.blocked;
  const inProgressCount = project.tasks.inProgress;
  const nextSteps = project.latestHandoff?.nextSteps ?? [];
  const totalTasks = project.tasks.pending + project.tasks.inProgress + project.tasks.completed + project.tasks.blocked + project.tasks.failed;
  const progressPct = totalTasks > 0
    ? Math.round((project.tasks.completed / totalTasks) * 100)
    : project.launchProgress;

  return (
    <button
      onClick={onClick}
      className={`group relative flex flex-col gap-2.5 rounded-lg border p-4 text-left transition-all hover:bg-white/[0.03] cursor-pointer ${theme.border} bg-[var(--color-surface)] ${
        project.health === "blocked" ? "bg-red-400/[0.02]" : project.health === "attention" ? "bg-amber-400/[0.02]" : ""
      }`}
    >
      {/* Row 1: Health dot + name + timestamp */}
      <div className="flex items-center gap-2">
        <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
          <span className={`h-2.5 w-2.5 rounded-full ${theme.dot}`} />
          {project.hasActiveSession && (
            <span className="absolute inset-0 animate-ping rounded-full bg-green-400/60" />
          )}
        </span>
        <span
          className={`text-sm font-semibold truncate ${
            project.health === "blocked" ? "text-red-400/90" : "text-white/90"
          }`}
        >
          {project.name}
        </span>
        <span className="ml-auto text-xs text-white/40 flex-shrink-0">
          {relativeTime(project.lastActive)}
        </span>
      </div>

      {/* Row 2: Sparkline + momentum + progress */}
      <div className="flex items-center gap-3">
        <Sparkline data={project.sparkline} color={sparkColor} width={100} height={24} />
        <span className={`text-[12px] font-mono ${mTheme.cls}`}>
          {mTheme.icon} {mTheme.label}
        </span>
        {progressPct > 0 && (
          <span className="ml-auto text-[12px] font-mono tabular-nums text-white/30">
            {progressPct}%
          </span>
        )}
      </div>

      {/* Row 3: Context line (blocker / next step / summary) */}
      {context && (
        <p className={`text-[13px] line-clamp-2 leading-relaxed ${CONTEXT_STYLE[context.type]}`}>
          {context.type === "blocker" && (
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-400 mr-1.5 relative top-[-1px]" />
          )}
          {context.type === "next" ? `Next: ${context.text}` : context.text}
        </p>
      )}

      {/* Row 4: Action-relevant pills */}
      <div className="flex flex-wrap items-center gap-1.5">
        {blockedCount > 0 && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-red-400/10 text-red-400/70">
            {blockedCount} blocked
          </span>
        )}
        {inProgressCount > 0 && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-amber-400/10 text-amber-400/70">
            {inProgressCount} in progress
          </span>
        )}
        {nextSteps.length > 1 && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-white/[0.05] text-white/40">
            {nextSteps.length - 1} more step{nextSteps.length - 1 !== 1 ? "s" : ""}
          </span>
        )}
        {project.hasActiveSession && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-green-400/10 text-green-400/70 ml-auto">
            agent working
          </span>
        )}
      </div>
    </button>
  );
}

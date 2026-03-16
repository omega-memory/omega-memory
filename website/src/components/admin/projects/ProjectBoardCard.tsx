import type { ProjectOverview } from "./types";
import { relativeTime, HEALTH_THEME, getContextLine, computeMomentum, MOMENTUM_THEME, getSparkColor } from "./utils";
import Sparkline from "./Sparkline";

interface ProjectBoardCardProps {
  project: ProjectOverview;
  onClick: () => void;
}

export default function ProjectBoardCard({ project, onClick }: ProjectBoardCardProps) {
  const theme = HEALTH_THEME[project.health];
  const context = getContextLine(project);
  const momentum = computeMomentum(project);
  const mTheme = MOMENTUM_THEME[momentum];
  const sparkColor = getSparkColor(momentum);

  const totalTasks = project.tasks.pending + project.tasks.inProgress + project.tasks.completed + project.tasks.blocked + project.tasks.failed;
  const progressPct = totalTasks > 0
    ? Math.round((project.tasks.completed / totalTasks) * 100)
    : project.launchProgress;
  const showProgress = progressPct > 0;

  const blockedCount = project.tasks.blocked;
  const inProgressCount = project.tasks.inProgress;

  const healthBg = project.health === "blocked" ? "bg-red-400/[0.03]"
    : project.health === "attention" ? "bg-amber-400/[0.03]"
    : "";

  return (
    <button
      onClick={onClick}
      data-project-id={project.id}
      className={`group w-full flex flex-col gap-2.5 rounded-lg border p-4 text-left transition-all hover:shadow-[var(--shadow-card-hover)] cursor-pointer ${theme.border} bg-[var(--color-surface)] ${healthBg}`}
    >
      {/* Row 1: Health dot + name + time + momentum */}
      <div className="flex items-center gap-2">
        <span className="relative flex h-2.5 w-2.5 shrink-0">
          <span className={`h-2.5 w-2.5 rounded-full ${theme.dot}`} />
          {project.hasActiveSession && (
            <span className="absolute inset-0 animate-ping rounded-full bg-green-400/60" />
          )}
        </span>
        <span className={`text-[14px] font-semibold truncate ${
          project.health === "blocked" ? "text-red-400/90" : "text-white/90"
        }`}>
          {project.name}
        </span>
        <span className="ml-auto flex items-center gap-1.5 shrink-0">
          <span className="text-[11px] text-white/40 font-mono">{relativeTime(project.lastActive)}</span>
          <span className={`text-[12px] font-mono ${mTheme.cls}`}>{mTheme.icon}</span>
        </span>
      </div>

      {/* Row 2: Sparkline + progress */}
      <div className="flex items-center gap-2.5">
        <Sparkline data={project.sparkline} color={sparkColor} width={60} height={20} />
        {showProgress && (
          <>
            <div className="flex-1 h-1 rounded-full bg-surface-elevated overflow-hidden">
              <div
                className={`h-full rounded-full ${progressPct >= 80 ? "bg-type-lesson/60" : progressPct >= 40 ? "bg-type-reminder/50" : "bg-ink-faint/30"}`}
                style={{ width: `${Math.max(progressPct, 3)}%` }}
              />
            </div>
            <span className="text-[11px] font-mono tabular-nums text-ink-faint">{progressPct}%</span>
          </>
        )}
      </div>

      {/* Row 3: Context line */}
      {context && (
        <p className={`text-[13px] line-clamp-2 leading-relaxed ${
          context.type === "blocker" ? "text-red-400/70" :
          context.type === "next" ? "text-amber-400/70" :
          "text-white/50"
        }`}>
          {context.type === "blocker" && (
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-400 mr-1.5 relative top-[-1px]" />
          )}
          {context.type === "next" ? `Next: ${context.text}` : context.text}
        </p>
      )}

      {/* Row 4: Action pills */}
      <div className="flex flex-wrap items-center gap-1.5">
        {blockedCount > 0 && (
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-red-400/10 text-red-400/70">
            {blockedCount} blocked
          </span>
        )}
        {inProgressCount > 0 && (
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-amber-400/10 text-amber-400/70">
            {inProgressCount} in progress
          </span>
        )}
        {project.hasActiveSession && (
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-green-400/10 text-green-400/70 ml-auto">
            agent working
          </span>
        )}
      </div>
    </button>
  );
}

import type { ProjectOverview } from "./types";
import { relativeTime, HEALTH_THEME, computeMomentum, MOMENTUM_THEME, getSparkColor } from "./utils";
import SlideDrawer from "../SlideDrawer";
import Sparkline from "./Sparkline";
import { StatCards, BlockersSection, NextStepsSection, DecisionsSection } from "./ProjectDetailSections";
import ActivityTimeline from "./ActivityTimeline";

interface ProjectDrawerProps {
  project: ProjectOverview | null;
  onClose: () => void;
  onOpenFull: (projectId: string) => void;
}

export default function ProjectDrawer({ project, onClose, onOpenFull }: ProjectDrawerProps) {
  if (!project) return null;

  const theme = HEALTH_THEME[project.health];
  const momentum = computeMomentum(project);
  const mTheme = MOMENTUM_THEME[momentum];
  const sparkColor = getSparkColor(momentum);
  const handoff = project.latestHandoff;

  return (
    <SlideDrawer open={!!project} onClose={onClose} title={project.name} width="xl">
      <div className="h-full overflow-y-auto p-5 space-y-5">
        {/* Open Full link */}
        <div className="flex justify-end">
          <button
            onClick={() => onOpenFull(project.id)}
            className="text-[13px] text-gold hover:text-gold-dim transition-colors font-mono"
          >
            Open Full &rarr;
          </button>
        </div>

        {/* Identity */}
        <div>
          <div className="flex items-center gap-2.5">
            <span className={`h-3 w-3 rounded-full ${theme.dot}`} />
            <h2 className="text-[20px] font-light text-ink tracking-tight">{project.name}</h2>
          </div>
          <p className="text-[14px] text-ink-faint mt-1">
            Last active {relativeTime(project.lastActive)}
            <span className="mx-1.5 text-ink-faint/30">&middot;</span>
            {project.sessionCount30d} sessions / 30d
            <span className="mx-1.5 text-ink-faint/30">&middot;</span>
            <span className={`font-mono ${mTheme.cls}`}>{mTheme.icon} {mTheme.label}</span>
          </p>
        </div>

        {/* Stat Cards */}
        <StatCards project={project} />

        {/* Full-width sparkline */}
        <div>
          <Sparkline data={project.sparkline} color={sparkColor} width={480} height={40} />
        </div>

        {/* Blockers */}
        {handoff?.blockedItems && <BlockersSection blockedItems={handoff.blockedItems} />}

        {/* Next Steps */}
        {handoff?.nextSteps && <NextStepsSection nextSteps={handoff.nextSteps} />}

        {/* Recent Work */}
        <ActivityTimeline activity={project.activity.slice(0, 8)} decisions={project.decisions.slice(0, 3)} />

        {/* Key Decisions */}
        <DecisionsSection decisions={project.decisions} limit={3} />
      </div>
    </SlideDrawer>
  );
}

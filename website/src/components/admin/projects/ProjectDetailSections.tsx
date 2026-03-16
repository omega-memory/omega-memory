import type { ProjectOverview, DecisionItem } from "./types";
import { relativeTime, humanize, humanizeDecision } from "./utils";

// ─── Stat Cards ──────────────────────────────────────────────

interface StatCardsProps {
  project: ProjectOverview;
}

export function StatCards({ project }: StatCardsProps) {
  const totalTasks = project.tasks.pending + project.tasks.inProgress + project.tasks.completed + project.tasks.blocked + project.tasks.failed;
  const hasTasks = totalTasks > 0;
  const handoff = project.latestHandoff;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
      {project.sessionCount30d > 0 && (
        <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
          <div className="text-[20px] font-light tabular-nums text-ink">{project.sessionCount30d}</div>
          <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Sessions</div>
          <div className="text-[11px] text-ink-faint/50">30 days</div>
        </div>
      )}
      {project.commitCount30d > 0 && (
        <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
          <div className="text-[20px] font-light tabular-nums text-ink">{project.commitCount30d}</div>
          <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Commits</div>
          <div className="text-[11px] text-ink-faint/50">30 days</div>
        </div>
      )}
      {project.decisionCount30d > 0 && (
        <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
          <div className="text-[20px] font-light tabular-nums text-ink">{project.decisionCount30d}</div>
          <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Decisions</div>
          <div className="text-[11px] text-ink-faint/50">30 days</div>
        </div>
      )}
      {hasTasks && (
        <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
          <div className="text-[20px] font-light tabular-nums text-ink">
            {project.tasks.completed}<span className="text-[14px] text-ink-faint">/{totalTasks}</span>
          </div>
          <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Tasks Done</div>
          {handoff?.blockedItems?.length ? (
            <div className="text-[11px] text-type-error">{handoff.blockedItems.length} blocked</div>
          ) : (
            <div className="text-[11px] text-ink-faint/50">{Math.round((project.tasks.completed / totalTasks) * 100)}%</div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Blockers Section ────────────────────────────────────────

interface BlockersSectionProps {
  blockedItems: string[];
}

export function BlockersSection({ blockedItems }: BlockersSectionProps) {
  if (blockedItems.length === 0) return null;
  return (
    <div>
      <span className="text-[13px] font-mono text-type-error uppercase tracking-wider">
        Blockers ({blockedItems.length})
      </span>
      <div className="mt-2 space-y-1.5">
        {blockedItems.map((item, i) => (
          <div key={i} className="flex items-start gap-2.5">
            <span className="text-type-error mt-0.5 shrink-0">!</span>
            <span className="text-[15px] text-type-error leading-snug">{humanize(item)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Next Steps Section ──────────────────────────────────────

interface NextStepsSectionProps {
  nextSteps: string[];
}

export function NextStepsSection({ nextSteps }: NextStepsSectionProps) {
  if (nextSteps.length === 0) return null;
  return (
    <div>
      <span className="text-[13px] font-mono text-type-reminder uppercase tracking-wider">
        Next Steps ({nextSteps.length})
      </span>
      <div className="mt-2 space-y-1.5">
        {nextSteps.map((step, i) => (
          <div key={i} className="flex items-start gap-2.5">
            <span className="text-type-reminder font-mono text-[14px] mt-0.5 shrink-0 w-5 text-right">{i + 1}.</span>
            <span className="text-[15px] text-ink-secondary leading-snug">{humanize(step)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Decisions Section ───────────────────────────────────────

interface DecisionsSectionProps {
  decisions: DecisionItem[];
  limit?: number;
}

export function DecisionsSection({ decisions, limit = 3 }: DecisionsSectionProps) {
  if (decisions.length === 0) return null;
  return (
    <div>
      <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
        Key Decisions
      </span>
      <div className="mt-2 space-y-2">
        {decisions.slice(0, limit).map((d) => (
          <div key={d.id} className="border-b border-edge-subtle/30 pb-2 last:border-0">
            <span className="text-[12px] text-ink-faint font-mono">{relativeTime(d.createdAt)}</span>
            <p className="text-[15px] text-ink-secondary leading-snug mt-0.5">{humanizeDecision(d.decision)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

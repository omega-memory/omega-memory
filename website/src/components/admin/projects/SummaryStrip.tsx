import type { ProjectOverview } from "./types";
import { computeMomentum, MOMENTUM_THEME } from "./utils";

interface SummaryStripProps {
  projects: ProjectOverview[];
  dormantCount: number;
  attentionCount: number;
}

export default function SummaryStrip({ projects, dormantCount, attentionCount }: SummaryStripProps) {
  const totalSessions = projects.reduce((s, p) => s + p.sessionCount30d, 0);
  const totalCommits = projects.reduce((s, p) => s + p.commitCount30d, 0);
  const totalPending = projects.reduce((s, p) => s + p.tasks.pending + p.tasks.inProgress, 0);
  const totalBlocked = projects.reduce((s, p) => s + p.tasks.blocked, 0);

  // Portfolio momentum: average of individual project momentum scores
  const momentumScores = projects.map((p) => {
    const m = computeMomentum(p);
    return m === "up" ? 2 : m === "steady" ? 1 : m === "cooling" ? -1 : -2;
  });
  const avgMomentum = momentumScores.length > 0 ? momentumScores.reduce((a, b) => a + b, 0) / momentumScores.length : 0;
  const portfolioMomentum = avgMomentum >= 1 ? "up" : avgMomentum >= 0 ? "steady" : avgMomentum >= -1 ? "cooling" : "stalled";
  const mTheme = MOMENTUM_THEME[portfolioMomentum as keyof typeof MOMENTUM_THEME];

  const stats = [
    { label: "Active", value: projects.length, sub: dormantCount > 0 ? `${dormantCount} dormant` : undefined, cls: "text-ink" },
    { label: "Blocked", value: totalBlocked, sub: attentionCount > 0 ? `${attentionCount} need attention` : undefined, cls: totalBlocked > 0 ? "text-type-error" : "text-ink" },
    { label: "Tasks Open", value: totalPending, sub: `${totalCommits} commits / 30d`, cls: totalPending > 0 ? "text-type-reminder" : "text-ink" },
    { label: "Momentum", value: mTheme.icon, sub: `${totalSessions} sessions / 30d`, cls: mTheme.cls, isIcon: true },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {stats.map((s) => (
        <div key={s.label} className="admin-card px-4 py-3.5">
          <div className={`text-[24px] font-light tabular-nums tracking-tight ${s.cls}`}>
            {s.isIcon ? <span className="text-[22px]">{s.value}</span> : s.value}
          </div>
          <div className="text-[13px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">{s.label}</div>
          {s.sub && <div className="text-[12px] text-ink-faint/60 mt-1">{s.sub}</div>}
        </div>
      ))}
    </div>
  );
}

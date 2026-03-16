import type { VerificationMetrics } from "../lib/types";

interface Props { verification?: VerificationMetrics; }

/** Inline tooltip: hover to see explanation. */
function Tip({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <span className="relative group/tip cursor-help">
      {children}
      <span className="pointer-events-none absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg bg-surface-elevated border border-border text-[12px] text-ink-secondary leading-snug w-56 opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150 shadow-lg">
        {text}
      </span>
    </span>
  );
}

const LAYER_TOOLTIPS: Record<string, string> = {
  sessions: "Percentage of agent sessions that wrote a handoff record on exit. Sessions without handoffs may have been short, cancelled, or crashed.",
  tasks: "Percentage of coordination tasks that reached 'completed' status. Pending and in-progress tasks are not counted as verified.",
  decisions: "Percentage of decisions that are still active and not superseded by a newer decision. 100% means no decision drift.",
  actions: "Percentage of external actions (emails, posts, deploys) that completed execution successfully.",
};

export default function RhoMetric({ verification }: Props) {
  const data = verification ?? null;
  if (!data || data.layers.length === 0) {
    return (
      <div className="space-y-2">
        <div className="admin-section-label">Verification Rate</div>
        <p className="text-[13px] text-ink-faint">No verification data available</p>
      </div>
    );
  }
  const { overallRho, layers } = data;
  const rhoColor = overallRho >= 70 ? "text-type-lesson" : overallRho >= 40 ? "text-type-reminder" : "text-type-error";
  const rhoBarColor = overallRho >= 70 ? "bg-type-lesson" : overallRho >= 40 ? "bg-type-reminder" : "bg-type-error";
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <Tip text="Overall completion rate across all agent coordination layers. Weighted by count: (total verified) / (total actions). Based on Catalini's Verified Network Scale.">
            <div className="admin-section-label border-b border-dashed border-ink-faint/30">Verification Rate</div>
          </Tip>
          <p className="text-ink-tertiary text-[13px] mt-0.5">
            <Tip text="Each layer tracks whether agent actions completed their lifecycle: sessions wrote handoffs, tasks reached completion, decisions remain active.">
              <span className="border-b border-dashed border-ink-faint/30">Verified agent actions / total actions (&#961;)</span>
            </Tip>
          </p>
        </div>
        <div className="text-right">
          <span className={`text-[28px] font-display font-bold tabular-nums ${rhoColor}`}>{overallRho}%</span>
          <div className="text-[11px] font-mono text-ink-faint tracking-wider uppercase">overall &#961;</div>
        </div>
      </div>
      <div className="h-2 rounded-full overflow-hidden bg-surface-elevated">
        <div className={`h-full rounded-full transition-all duration-700 ${rhoBarColor}`} style={{ width: `${overallRho}%` }} />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
        {layers.map((layer) => {
          const lc = layer.rho >= 70 ? "text-type-lesson" : layer.rho >= 40 ? "text-type-reminder" : "text-type-error";
          const lb = layer.rho >= 70 ? "bg-type-lesson" : layer.rho >= 40 ? "bg-type-reminder" : "bg-type-error";
          const tooltip = LAYER_TOOLTIPS[layer.label.toLowerCase()] ?? `${layer.label}: ${layer.verified} verified out of ${layer.total} total.`;
          return (
            <Tip key={layer.label} text={tooltip}>
              <div className="p-2.5 rounded-lg bg-surface/50">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[12px] font-mono text-ink-tertiary tracking-wider uppercase border-b border-dashed border-ink-faint/30">{layer.label}</span>
                  <span className={`text-[14px] font-bold tabular-nums ${lc}`}>{layer.rho}%</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden bg-surface-elevated">
                  <div className={`h-full rounded-full transition-all duration-500 ${lb}`} style={{ width: `${layer.rho}%` }} />
                </div>
                <div className="text-[11px] text-ink-faint mt-1 tabular-nums">{layer.verified}/{layer.total}</div>
              </div>
            </Tip>
          );
        })}
      </div>
    </div>
  );
}

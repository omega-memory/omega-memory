import { priorityColor, priorityBg, priorityPercent, priorityBarColor } from "./helpers";

export default function ScoreBadge({
  impact,
  feasibility,
  priority,
}: {
  impact: number | null;
  feasibility: number | null;
  priority: number;
}) {
  return (
    <div className="flex items-center gap-3">
      {/* Visual priority bar */}
      <div className="flex items-center gap-2.5 flex-1 min-w-0">
        <span
          className={`text-[15px] font-semibold tabular-nums ${priorityColor(priority)}`}
        >
          {priority}/25
        </span>
        <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${priorityBarColor(priority)}`}
            style={{ width: `${priorityPercent(priority)}%` }}
          />
        </div>
      </div>

      {/* Impact & Feasibility labels */}
      {impact !== null && feasibility !== null && (
        <div className="flex items-center gap-3 text-[14px] text-ink-faint shrink-0">
          <span>
            Impact <strong className="text-ink-secondary">{impact}</strong>
          </span>
          <span>
            Feasibility <strong className="text-ink-secondary">{feasibility}</strong>
          </span>
        </div>
      )}
    </div>
  );
}

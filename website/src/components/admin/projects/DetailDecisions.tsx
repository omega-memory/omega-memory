import type { DecisionItem } from "./types";
import { relativeTime } from "./utils";

interface DetailDecisionsProps {
  decisions: DecisionItem[];
}

export default function DetailDecisions({ decisions }: DetailDecisionsProps) {
  if (decisions.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
        Decisions ({decisions.length})
      </h3>
      <div className="space-y-2 max-h-56 overflow-y-auto">
        {decisions.map((d) => (
          <div key={d.id} className="rounded px-2.5 py-2 bg-white/[0.02]">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-type-decision)]/15 text-[var(--color-type-decision)]/70 font-medium">
                {d.domain}
              </span>
              <span className="text-[10px] text-white/20 ml-auto">
                {relativeTime(d.createdAt)}
              </span>
            </div>
            <p className="text-xs text-white/60 leading-relaxed">
              {d.decision}
            </p>
            {d.rationale && (
              <p className="text-xs text-white/30 italic mt-1 leading-relaxed">
                {d.rationale.length > 150
                  ? d.rationale.slice(0, 150) + "..."
                  : d.rationale}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

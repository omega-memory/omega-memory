import type { DecisionItem } from "./types";
import { relativeTime } from "./utils";

interface DecisionTimelineProps {
  decisions: DecisionItem[];
}

const DOMAIN_COLORS: Record<string, string> = {
  general: "bg-blue-400/20 text-blue-400/80 border-blue-400/30",
  architecture: "bg-teal-400/20 text-teal-400/80 border-teal-400/30",
  implementation: "bg-green-400/20 text-green-400/80 border-green-400/30",
  testing: "bg-amber-400/20 text-amber-400/80 border-amber-400/30",
  deployment: "bg-red-400/20 text-red-400/80 border-red-400/30",
  design: "bg-purple-400/20 text-purple-400/80 border-purple-400/30",
};

const DEFAULT_DOMAIN_COLOR = "bg-white/[0.06] text-white/40 border-white/[0.08]";

export default function DecisionTimeline({
  decisions,
}: DecisionTimelineProps) {
  if (decisions.length === 0) return null;

  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-4">
        Decision Timeline ({decisions.length})
      </h3>
      <div className="relative">
        <div className="absolute left-[7px] top-2 bottom-2 w-px bg-white/[0.08]" />
        <div className="space-y-3">
          {decisions.map((d) => {
            const domainColor = DOMAIN_COLORS[d.domain] ?? DEFAULT_DOMAIN_COLOR;
            return (
              <div
                key={d.id}
                className="relative flex items-start gap-3 pl-5 w-full text-left group hover:bg-white/[0.02] rounded-lg py-1.5 pr-2 transition-colors"
              >
                <span className="absolute left-[4px] top-3 h-[7px] w-[7px] rounded-full bg-white/20 ring-2 ring-[var(--color-canvas)] group-hover:bg-white/40 transition-colors" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${domainColor}`}>
                      {d.domain}
                    </span>
                    <span className="text-xs text-white/20 ml-auto flex-shrink-0">
                      {relativeTime(d.createdAt)}
                    </span>
                  </div>
                  <p className="text-xs text-white/55 leading-relaxed line-clamp-2">
                    {d.decision}
                  </p>
                  {d.rationale && (
                    <p className="text-[11px] text-white/40 leading-relaxed mt-0.5 line-clamp-1 italic">
                      {d.rationale}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

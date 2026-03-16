import type { ActivityItem } from "./types";
import { relativeTime } from "./utils";

interface DetailActivityProps {
  activity: ActivityItem[];
}

const ICON_MAP: Record<string, { icon: string; color: string }> = {
  session: { icon: "S", color: "bg-amber-400/20 text-amber-400/80" },
  commit: { icon: "C", color: "bg-green-400/20 text-green-400/80" },
  decision: { icon: "D", color: "bg-blue-400/20 text-blue-400/80" },
  task: { icon: "T", color: "bg-purple-400/20 text-purple-400/80" },
};

export default function DetailActivity({ activity }: DetailActivityProps) {
  if (activity.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
        Recent Activity
      </h3>
      <div className="space-y-0 max-h-64 overflow-y-auto">
        {activity.map((item, i) => {
          const config = ICON_MAP[item.type] ?? ICON_MAP.session;
          return (
            <div
              key={`${item.type}-${i}`}
              className="flex items-start gap-2.5 py-1.5 relative"
            >
              {/* Timeline connector line */}
              {i < activity.length - 1 && (
                <div className="absolute left-[9px] top-[22px] w-px h-[calc(100%-8px)] bg-white/[0.06]" />
              )}
              {/* Icon */}
              <span
                className={`h-[18px] w-[18px] rounded flex items-center justify-center text-[9px] font-bold flex-shrink-0 z-10 ${config.color}`}
              >
                {config.icon}
              </span>
              {/* Content */}
              <div className="flex-1 min-w-0">
                <p className="text-xs text-white/60 truncate">{item.summary}</p>
              </div>
              {/* Timestamp */}
              <span className="text-xs text-white/20 flex-shrink-0 pt-0.5">
                {relativeTime(item.timestamp)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

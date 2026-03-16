import { useMemo, useState } from "react";
import type { SessionHeatmapEntry } from "./types";

interface SessionHeatmapProps {
  data: SessionHeatmapEntry[];
}

const CELL_SIZE = 18;
const CELL_GAP = 3;
const LABEL_WIDTH = 32;
const HOURS = [0, 3, 6, 9, 12, 15, 18, 21];
const HOUR_LABELS = ["12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p"];

export default function SessionHeatmap({ data }: SessionHeatmapProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  const { dates, maxCount, grid } = useMemo(() => {
    const dates: string[] = [];
    const now = new Date();
    for (let i = 13; i >= 0; i--) {
      const d = new Date(now.getTime() - i * 86400000);
      dates.push(d.toISOString().slice(0, 10));
    }

    const grid = new Map<string, number>();
    let maxCount = 1;
    for (const entry of data) {
      const key = `${entry.date}:${entry.hour}`;
      grid.set(key, entry.count);
      if (entry.count > maxCount) maxCount = entry.count;
    }

    return { dates, maxCount, grid };
  }, [data]);

  const width = LABEL_WIDTH + dates.length * (CELL_SIZE + CELL_GAP);
  const height = 20 + HOURS.length * (CELL_SIZE + CELL_GAP);

  function cellColor(count: number): string {
    if (count === 0) return "rgba(255,255,255,0.03)";
    const intensity = Math.min(count / maxCount, 1);
    const alpha = 0.15 + intensity * 0.55;
    return `rgba(20, 184, 166, ${alpha})`;
  }

  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-3">
        Session Activity (14 days)
      </h3>
      <div className="overflow-x-auto">
        <svg width={width} height={height} className="block">
          {HOURS.map((hour, hi) => (
            <text
              key={hour}
              x={LABEL_WIDTH - 4}
              y={20 + hi * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2 + 4}
              textAnchor="end"
              className="fill-white/20 text-[9px]"
              style={{ fontFamily: "ui-monospace, monospace" }}
            >
              {HOUR_LABELS[hi]}
            </text>
          ))}

          {dates.map((date, di) =>
            HOURS.map((hour, hi) => {
              const count = grid.get(`${date}:${hour}`) ?? 0;
              const x = LABEL_WIDTH + di * (CELL_SIZE + CELL_GAP);
              const y = 20 + hi * (CELL_SIZE + CELL_GAP);
              return (
                <rect
                  key={`${date}-${hour}`}
                  x={x}
                  y={y}
                  width={CELL_SIZE}
                  height={CELL_SIZE}
                  rx={3}
                  fill={cellColor(count)}
                  stroke="rgba(255,255,255,0.04)"
                  strokeWidth={0.5}
                  className="transition-colors"
                  onMouseEnter={(e) => {
                    const dayLabel = new Date(date + "T00:00:00Z").toLocaleDateString("en-US", {
                      weekday: "short",
                      month: "short",
                      day: "numeric",
                    });
                    setTooltip({
                      x: e.clientX,
                      y: e.clientY,
                      text: `${dayLabel} ${HOUR_LABELS[hi]}: ${count} session${count !== 1 ? "s" : ""}`,
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
              );
            }),
          )}

          {dates.map((date, di) => {
            if (di % 2 !== 0) return null;
            const d = new Date(date + "T00:00:00Z");
            const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
            return (
              <text
                key={`label-${date}`}
                x={LABEL_WIDTH + di * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2}
                y={12}
                textAnchor="middle"
                className="fill-white/15 text-[8px]"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {label}
              </text>
            );
          })}
        </svg>
      </div>

      {tooltip && (
        <div
          className="fixed z-[100] rounded px-2 py-1 bg-black/90 text-[10px] text-white/80 pointer-events-none border border-white/[0.08]"
          style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

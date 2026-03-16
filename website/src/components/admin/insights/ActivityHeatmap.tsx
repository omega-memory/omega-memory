import React from "react";
import type { HeatmapCell } from "../lib/types";
import { TYPE_REGISTRY } from "../contentUtils";
import Tooltip from "./Tooltip";
import CollapsibleSection from "./CollapsibleSection";

export default function ActivityHeatmap({ heatmap }: { heatmap: Record<string, HeatmapCell> }) {
  const weeks = 12;
  const today = new Date();
  const cells: { date: string; cell: HeatmapCell; dayOfWeek: number }[] = [];
  const emptyCell: HeatmapCell = { total: 0, types: {}, projects: {} };

  for (let i = weeks * 7 - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    cells.push({
      date: key,
      cell: heatmap[key] || emptyCell,
      dayOfWeek: d.getDay(),
    });
  }

  const max = Math.max(1, ...cells.map((c) => c.cell.total));

  function intensity(count: number): string {
    if (count === 0) return "bg-surface-elevated";
    const ratio = count / max;
    if (ratio < 0.25) return "bg-type-decision/20";
    if (ratio < 0.5) return "bg-type-decision/40";
    if (ratio < 0.75) return "bg-type-decision/60";
    return "bg-type-decision/90";
  }

  const columns: typeof cells[] = [];
  for (let w = 0; w < weeks; w++) {
    columns.push(cells.slice(w * 7, (w + 1) * 7));
  }

  const activeDays = cells.filter((c) => c.cell.total > 0);
  const totalMems = cells.reduce((a, c) => a + c.cell.total, 0);
  const avgPerDay = activeDays.length > 0 ? Math.round(totalMems / activeDays.length) : 0;

  let busiestDay = "";
  let busiestCount = 0;
  for (const c of cells) {
    if (c.cell.total > busiestCount) {
      busiestCount = c.cell.total;
      busiestDay = c.date;
    }
  }
  const busiestLabel = busiestDay
    ? new Date(busiestDay + "T12:00:00").toLocaleDateString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
      })
    : "";

  let streak = 0;
  for (let i = cells.length - 1; i >= 0; i--) {
    if (cells[i].cell.total > 0) {
      streak++;
    } else {
      break;
    }
  }
  let lastActiveAgo = 0;
  if (streak === 0) {
    for (let i = cells.length - 1; i >= 0; i--) {
      if (cells[i].cell.total > 0) break;
      lastActiveAgo++;
    }
  }

  const monthLabels: { col: number; label: string }[] = [];
  let prevMonth = -1;
  for (let ci = 0; ci < columns.length; ci++) {
    const firstCell = columns[ci][0];
    if (firstCell) {
      const month = new Date(firstCell.date + "T12:00:00").getMonth();
      if (month !== prevMonth) {
        monthLabels.push({
          col: ci,
          label: new Date(firstCell.date + "T12:00:00").toLocaleDateString("en-US", { month: "short" }),
        });
        prevMonth = month;
      }
    }
  }

  const dayLabels = ["", "M", "", "W", "", "F", ""];
  const gridCols = `28px repeat(${weeks}, 1fr)`;

  const summaryText = `${avgPerDay}/day avg${streak > 0 ? `, ${streak}d streak` : ""}`;

  return (
    <CollapsibleSection label="Activity" summary={summaryText}>
      <div className="flex items-center gap-4 mb-3 text-[16px] text-ink-tertiary flex-wrap">
        <span>
          Avg <span className="text-ink tabular-nums font-medium">{avgPerDay}</span>/day
        </span>
        {busiestDay && (
          <span>
            Busiest: <span className="text-ink">{busiestLabel}</span>{" "}
            <span className="tabular-nums">({busiestCount})</span>
          </span>
        )}
        {streak > 0 ? (
          <span>
            Streak: <span className="text-type-lesson tabular-nums font-medium">{streak}d</span>
          </span>
        ) : lastActiveAgo > 0 ? (
          <span>
            Last active: <span className="tabular-nums">{lastActiveAgo}d ago</span>
          </span>
        ) : null}
      </div>

      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: gridCols, gridTemplateRows: `repeat(7, 1fr)` }}
      >
        {Array.from({ length: 7 }, (_, row) => (
          <React.Fragment key={row}>
            <div className="flex items-center justify-end pr-1">
              <span className="text-[14px] text-ink-tertiary leading-tight">{dayLabels[row]}</span>
            </div>
            {columns.map((col, wi) => {
              const c = col[row];
              if (!c) return <div key={wi} />;

              const topTypes = Object.entries(c.cell.types as Record<string, number>)
                .sort((a, b) => b[1] - a[1]);
              const shownTypes = topTypes.slice(0, 3);
              const remainingTypes = topTypes.length - 3;
              const projectNames = Object.keys(c.cell.projects).join(", ");
              const dateLabel = new Date(c.date + "T12:00:00").toLocaleDateString("en-US", {
                weekday: "short",
                month: "short",
                day: "numeric",
              });

              return (
                <Tooltip
                  key={c.date}
                  content={
                    c.cell.total === 0 ? (
                      <span className="text-ink-tertiary">{dateLabel}: no activity</span>
                    ) : (
                      <div className="space-y-1 min-w-[120px]">
                        <div className="text-ink font-medium">{dateLabel}</div>
                        <div className="border-t border-edge-strong" />
                        <div className="text-ink">
                          <span className="font-semibold tabular-nums">{c.cell.total}</span> memories
                        </div>
                        {shownTypes.map(([type, count]) => (
                          <div key={type} className="flex items-center gap-1.5">
                            <span
                              className="w-1.5 h-1.5 rounded-full shrink-0"
                              style={{ backgroundColor: TYPE_REGISTRY[type]?.color || TYPE_REGISTRY.unknown.color }}
                              aria-hidden="true"
                            />
                            <span className="text-ink-secondary">
                              {TYPE_REGISTRY[type]?.label || type}
                            </span>
                            <span className="tabular-nums ml-auto">{count}</span>
                          </div>
                        ))}
                        {remainingTypes > 0 && (
                          <div className="text-ink-tertiary">+{remainingTypes} more</div>
                        )}
                        {projectNames && (
                          <div className="text-ink-tertiary border-t border-edge-strong pt-1 mt-1">
                            {projectNames}
                          </div>
                        )}
                      </div>
                    )
                  }
                >
                  <div
                    className={`aspect-square min-w-[14px] min-h-[14px] rounded-[3px] ${intensity(c.cell.total)} transition-colors hover:ring-1 hover:ring-type-decision/30`}
                    tabIndex={0}
                    role="gridcell"
                    aria-label={`${dateLabel}: ${c.cell.total} memories`}
                  />
                </Tooltip>
              );
            })}
          </React.Fragment>
        ))}
      </div>

      <div
        className="grid gap-[3px] mt-1"
        style={{ gridTemplateColumns: gridCols }}
      >
        <div />
        {columns.map((_, wi) => {
          const ml = monthLabels.find((m) => m.col === wi);
          return (
            <div key={wi} className="min-w-0">
              {ml && (
                <span className="text-[14px] text-ink-tertiary leading-tight">{ml.label}</span>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-1.5 mt-2">
        <span className="text-[14px] text-ink-faint">Less</span>
        {["bg-surface-elevated", "bg-type-decision/20", "bg-type-decision/40", "bg-type-decision/60", "bg-type-decision/90"].map(
          (cls, i) => (
            <div key={i} className={`w-[14px] h-[14px] rounded-[2px] ${cls}`} />
          ),
        )}
        <span className="text-[14px] text-ink-faint">More</span>
      </div>
    </CollapsibleSection>
  );
}

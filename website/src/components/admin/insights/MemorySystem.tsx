import React from "react";
import {
  AreaChart,
  Area,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { InsightsData } from "../lib/types";
import AdminTooltip from "./AdminTooltip";

export default function MemorySystem({
  memory,
  period,
}: {
  memory: InsightsData["memory"];
  period: number;
}) {
  const today = new Date();
  const data: { date: string; label: string; count: number }[] = [];
  for (let i = period - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const iso = d.toISOString().slice(0, 10);
    data.push({
      date: iso,
      label: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      count: memory.byDay[iso] || 0,
    });
  }

  const dailyTotal = Object.values(memory.byDay).reduce((a, b) => a + b, 0);

  return (
    <div className="p-4 admin-card space-y-5">
      <h4 className="admin-section-label">
        Memory System
      </h4>

      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[16px] text-ink-tertiary">Memories / day</span>
          <span className="text-[16px] text-ink-tertiary font-mono tabular-nums">
            {dailyTotal} in period{memory.totalCloud ? ` / ${memory.totalCloud} cloud` : ""}
          </span>
        </div>
        <ResponsiveContainer width="100%" height={60}>
          <AreaChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-type-decision)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--color-type-decision)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <Tooltip
              content={
                <AdminTooltip
                  formatLabel={(label) => {
                    const point = data.find((d) => d.date === label || d.label === label);
                    return point?.label || label;
                  }}
                />
              }
              cursor={{ stroke: "var(--color-type-decision)", strokeWidth: 1, opacity: 0.4 }}
            />
            <Area
              type="monotone"
              dataKey="count"
              name="memories"
              stroke="var(--color-type-decision)"
              strokeWidth={2}
              fill="url(#chartGradient)"
              fillOpacity={1}
              dot={false}
              activeDot={{
                r: 3,
                fill: "var(--color-type-decision)",
                stroke: "var(--color-surface)",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

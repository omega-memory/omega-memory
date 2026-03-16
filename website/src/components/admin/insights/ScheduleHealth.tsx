import React from "react";
import type { ScheduleInfo } from "../lib/types";
import { timeAgo, humanSchedule, statusDot } from "../contentUtils";
import CollapsibleSection from "./CollapsibleSection";

export default function ScheduleHealth({ schedules }: { schedules: ScheduleInfo[] }) {
  const problems = schedules.filter(
    (s) => s.lastStatus === "error" || s.overdue,
  );
  const healthy = schedules.length - problems.length;
  const allHealthy = problems.length === 0;

  const mostRecentRun = schedules.reduce<string | null>((latest, s) => {
    if (!s.lastRunAt) return latest;
    if (!latest) return s.lastRunAt;
    return s.lastRunAt > latest ? s.lastRunAt : latest;
  }, null);

  const summaryText = schedules.length === 0
    ? "No jobs configured"
    : allHealthy
      ? `All ${schedules.length} healthy`
      : `${problems.length} problem${problems.length !== 1 ? "s" : ""}`;

  return (
    <CollapsibleSection label="Schedule Health" summary={summaryText} defaultOpen={problems.length > 0}>
      {schedules.length === 0 ? (
        <p className="text-[16px] text-ink-tertiary">No jobs configured</p>
      ) : allHealthy ? (
        <div className="flex items-center gap-2 text-[16px]">
          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${statusDot("ok")}`} />
          <span className="text-ink">
            All {schedules.length} jobs healthy
          </span>
          {mostRecentRun && (
            <span className="text-ink-tertiary ml-auto text-[16px] tabular-nums">
              Last run: {timeAgo(mostRecentRun)}
            </span>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          <div className="text-[16px] text-ink-tertiary mb-2">
            <span className="text-type-lesson tabular-nums">{healthy}</span> healthy,{" "}
            <span className="text-type-error tabular-nums">{problems.length}</span> problem{problems.length !== 1 ? "s" : ""}
          </div>
          <div className="space-y-1">
            {problems.map((s) => {
              const isError = s.lastStatus === "error";
              const bgClass = isError
                ? "bg-type-error/[0.04] border-l-2 border-type-error"
                : "bg-type-reminder/[0.04] border-l-2 border-type-reminder";
              const dotStatus = isError ? "error" : "overdue";

              let issue: string;
              if (isError) {
                issue = "Error";
              } else if (!s.lastRunAt) {
                issue = "Never ran";
              } else {
                const ago = timeAgo(s.lastRunAt);
                const expected = humanSchedule(s).toLowerCase();
                issue = `Last run ${ago} (expected ${expected})`;
              }

              return (
                <div
                  key={s.id}
                  className={`flex items-center gap-2 py-2 px-3 rounded-lg text-[16px] ${bgClass}`}
                >
                  <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${statusDot(dotStatus)}`} />
                  <span className="font-medium text-ink truncate">{s.name}</span>
                  <span className="text-ink-tertiary truncate ml-auto text-[16px]">
                    {issue}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </CollapsibleSection>
  );
}

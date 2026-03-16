import {
  type Schedule,
  computeStatus,
  computeNextRun,
  humanSchedule,
  relativeTime,
} from "./jobUtils";
import { StatusPill, ScheduleChip, ToggleSwitch, OwnerBadge } from "./atoms";

interface JobRowProps {
  schedule: Schedule;
  selected: boolean;
  onSelect: (id: string) => void;
  onToggle: (id: string, enabled: boolean) => void;
  healthDots?: string[];
  isFocused?: boolean;
  isChecked?: boolean;
}

export function JobRow({ schedule, selected, onSelect, onToggle, healthDots, isFocused, isChecked }: JobRowProps) {
  const status = computeStatus(schedule, healthDots);
  const nextRun = computeNextRun(schedule);
  const isIssue = status === "error" || status === "late";

  return (
    <button
      type="button"
      data-job-id={schedule.id}
      onClick={() => onSelect(schedule.id)}
      aria-label={`${schedule.name} - ${status}`}
      className={`
        w-full text-left px-5 py-4 transition-colors duration-150
        border-b border-white/[0.04] min-h-[64px]
        ${isIssue ? "border-l-[3px]" : "border-l-[3px] border-l-transparent"}
        ${status === "error" ? "border-l-type-error" : ""}
        ${status === "late" ? "border-l-type-reminder" : ""}
        ${selected ? "bg-gold/[0.06]" : "hover:bg-white/[0.025]"}
        ${isFocused ? "ring-1 ring-gold/40 bg-gold/[0.03]" : ""}
        ${isChecked ? "bg-gold/[0.08]" : ""}
      `}
    >
      {/* Row grid: status | health | name+desc | schedule | last | next | toggle */}
      <div className="grid grid-cols-[84px_56px_1fr_140px_72px_72px_44px] items-center gap-3">
        {/* Status */}
        <div className="flex items-center">
          <StatusPill status={status} />
        </div>

        {/* Health dots */}
        <div className="flex items-center gap-0.5">
          {healthDots && healthDots.length > 0 ? (
            healthDots.map((s, i) => (
              <span
                key={i}
                className={`w-1.5 h-1.5 rounded-full ${
                  s === "ok" ? "bg-type-lesson" : s === "error" ? "bg-type-error" : "bg-ink-faint/30"
                }`}
              />
            ))
          ) : (
            <span className="text-[11px] text-ink-faint/40">--</span>
          )}
        </div>

        {/* Name + Description */}
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[18px] font-medium text-ink truncate leading-snug">
              {schedule.name}
            </span>
            <OwnerBadge label={schedule.label} />
          </div>
          {schedule.description && (
            <p className="text-[16px] text-ink-tertiary truncate mt-0.5 leading-relaxed">
              {schedule.description}
            </p>
          )}
        </div>

        {/* Schedule */}
        <div className="flex justify-start">
          <ScheduleChip text={humanSchedule(schedule)} />
        </div>

        {/* Last run */}
        <div className="text-[15px] text-ink-tertiary font-mono tabular-nums">
          {relativeTime(schedule.last_run_at)}
        </div>

        {/* Next run */}
        <div className="text-[15px] text-ink-tertiary font-mono tabular-nums">
          {nextRun ? relativeTime(nextRun) : "--"}
        </div>

        {/* Toggle */}
        <div className="flex justify-end">
          <ToggleSwitch
            checked={schedule.enabled}
            onChange={(checked) => onToggle(schedule.id, checked)}
          />
        </div>
      </div>
    </button>
  );
}

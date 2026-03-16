import React, { useState, useEffect, useCallback } from "react";
import {
  type Schedule,
  computeNextRun,
  humanSchedule,
  ownerBadge,
} from "../jobs/jobUtils";

interface Props {
  schedules: Schedule[];
  onSkipToggle: (id: string, skipNext: boolean) => Promise<void>;
}

/** Format a millisecond countdown into human-friendly text */
function formatCountdown(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;

  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

/** Extract just the bg-* class from ownerBadge cls string for the dot */
function extractBgClass(cls: string): string {
  const match = cls.match(/bg-[^\s]+/);
  return match ? match[0] : "bg-ink-tertiary";
}

const TWO_HOURS_MS = 2 * 60 * 60 * 1000;
const FIFTEEN_MIN_MS = 15 * 60 * 1000;
const FIVE_MIN_MS = 5 * 60 * 1000;
const TWO_MIN_MS = 2 * 60 * 1000;
const JUST_RAN_MS = 5 * 60 * 1000;

interface TriggerRow {
  schedule: Schedule;
  nextRun: Date;
}

export default function UpcomingTriggers({ schedules, onSkipToggle }: Props) {
  const [now, setNow] = useState(() => Date.now());
  const [expanded, setExpanded] = useState(false);
  const [toggling, setToggling] = useState<string | null>(null);

  // Tick every second
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // Build sorted trigger rows from enabled schedules
  const rows: TriggerRow[] = [];
  for (const s of schedules) {
    if (!s.enabled) continue;
    const next = computeNextRun(s);
    if (!next) continue;
    rows.push({ schedule: s, nextRun: next });
  }
  rows.sort((a, b) => a.nextRun.getTime() - b.nextRun.getTime());

  if (rows.length === 0) return null;

  // Split into within-2h and beyond
  const within2h = rows.filter(
    (r) => r.nextRun.getTime() - now <= TWO_HOURS_MS,
  );
  const beyond2h = rows.filter(
    (r) => r.nextRun.getTime() - now > TWO_HOURS_MS,
  );

  const visible = expanded ? rows : within2h;
  const hiddenCount = beyond2h.length;

  const handleToggle = useCallback(
    async (id: string, skipNext: boolean) => {
      setToggling(id);
      try {
        await onSkipToggle(id, skipNext);
      } finally {
        setToggling(null);
      }
    },
    [onSkipToggle],
  );

  if (visible.length === 0 && hiddenCount === 0) return null;

  return (
    <div className="rounded-xl border border-edge bg-surface-elevated/50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-edge">
        <h4 className="text-[13px] font-semibold text-ink tracking-tight">
          Upcoming Triggers
        </h4>
        <span className="text-[12px] text-ink-tertiary font-mono tabular-nums">
          {rows.length} in queue
        </span>
      </div>

      {/* Rows */}
      <div className="divide-y divide-edge/50">
        {visible.map((r) => {
          const { schedule: s, nextRun } = r;
          const ms = nextRun.getTime() - now;
          const badge = ownerBadge(s.label);
          const dotBg = extractBgClass(badge.cls);

          // Check if "just ran"
          const justRan =
            s.last_run_at &&
            now - new Date(s.last_run_at).getTime() < JUST_RAN_MS &&
            now - new Date(s.last_run_at).getTime() >= 0;

          // Countdown color
          let countdownCls = "text-ink-secondary";
          if (ms < FIVE_MIN_MS) countdownCls = "text-type-error";
          else if (ms < FIFTEEN_MIN_MS) countdownCls = "text-type-reminder";

          // Prefix with ~ when < 2min
          const prefix = ms > 0 && ms < TWO_MIN_MS ? "~" : "";

          const isSkipping = s.skip_next;
          const isToggling = toggling === s.id;

          return (
            <div
              key={s.id}
              className={`flex items-center gap-3 px-4 py-2.5 transition-opacity ${
                isSkipping ? "opacity-50" : ""
              }`}
            >
              {/* Owner dot */}
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${dotBg}`}
              />

              {/* Name + schedule */}
              <div className="flex-1 min-w-0">
                <span className="text-[13px] text-ink font-medium truncate block">
                  {s.name}
                </span>
                <span className="text-[11px] text-ink-tertiary truncate block">
                  {humanSchedule(s)}
                </span>
              </div>

              {/* Countdown / status */}
              <div className="shrink-0 text-right">
                {isSkipping ? (
                  <span className="text-[13px] text-ink-tertiary font-mono tabular-nums">
                    Skipping
                  </span>
                ) : justRan ? (
                  <span className="text-[13px] text-type-lesson font-mono tabular-nums">
                    Just ran
                  </span>
                ) : (
                  <span
                    className={`text-[13px] font-mono tabular-nums ${countdownCls}`}
                  >
                    {prefix}
                    {formatCountdown(ms)}
                  </span>
                )}
              </div>

              {/* Skip / Undo button */}
              <button
                onClick={() => handleToggle(s.id, !s.skip_next)}
                disabled={isToggling}
                className="shrink-0 text-[11px] font-medium px-2 py-1 rounded-md bg-surface-hover text-ink-secondary hover:text-ink hover:bg-surface-elevated transition-colors disabled:opacity-50"
              >
                {isToggling ? "..." : isSkipping ? "Undo" : "Skip"}
              </button>
            </div>
          );
        })}
      </div>

      {/* Show more expander */}
      {hiddenCount > 0 && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="w-full py-2.5 text-[12px] text-ink-tertiary hover:text-ink-secondary transition-colors border-t border-edge/50"
        >
          Show {hiddenCount} more
        </button>
      )}
      {expanded && hiddenCount > 0 && (
        <button
          onClick={() => setExpanded(false)}
          className="w-full py-2.5 text-[12px] text-ink-tertiary hover:text-ink-secondary transition-colors border-t border-edge/50"
        >
          Show less
        </button>
      )}
    </div>
  );
}

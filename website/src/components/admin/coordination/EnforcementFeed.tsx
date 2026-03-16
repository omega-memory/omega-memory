import { useMemo } from "react";
import type { CoordinationMetric } from "../lib/types";
import { formatTimeAgo } from "../lib/coordination-utils";

/** Metric names that represent enforcement events. */
const ENFORCEMENT_METRICS = new Set([
  "conflict_blocked_by_guard",
  "conflict_detected",
  "conflict_force_override",
  "gate_check_medium",
  "gate_check_high",
  "deadlock_cycle",
]);

interface MetricDisplay {
  label: string;
  dotColor: string;
}

const METRIC_DISPLAY: Record<string, MetricDisplay> = {
  conflict_blocked_by_guard: { label: "Guard block", dotColor: "bg-red-400" },
  conflict_detected: { label: "Conflict", dotColor: "bg-amber-400" },
  conflict_force_override: { label: "Force override", dotColor: "bg-amber-400" },
  gate_check_medium: { label: "Gate check", dotColor: "bg-amber-400" },
  gate_check_high: { label: "Gate check (high)", dotColor: "bg-red-400" },
  deadlock_cycle: { label: "Deadlock", dotColor: "bg-red-400" },
};

function parseMetadata(raw: string | null): Record<string, unknown> {
  if (!raw) return {};
  try { return JSON.parse(raw); } catch { return {}; }
}

function extractDetail(metric: CoordinationMetric): string {
  const meta = parseMetadata(metric.metadata);
  const file = meta.file as string | undefined;
  const previousOwner = meta.previous_owner as string | undefined;
  const claimedBy = meta.claimed_by as string | undefined;
  const cycle = meta.cycle as string[] | undefined;
  const action = meta.action as string | undefined;
  const branch = meta.branch as string | undefined;
  const sessionId = meta.session_id as string | undefined;
  const agentName = meta.agent_name as string | undefined;

  if (file) {
    const parts = file.split("/");
    const short = parts.length > 3 ? `.../${parts.slice(-2).join("/")}` : file;
    const owner = previousOwner || claimedBy;
    return owner ? `${short} (${owner.slice(0, 8)})` : short;
  }
  if (cycle) return `${cycle.length} agents`;

  // Gate checks: show action + branch/agent for context
  if (action) {
    const parts = [action];
    if (branch) parts.push(branch);
    else if (agentName) parts.push(agentName);
    else if (sessionId) parts.push(sessionId.slice(0, 8));
    return parts.join(" · ");
  }
  return "";
}

interface EnforcementFeedProps {
  metrics: CoordinationMetric[];
}

export default function EnforcementFeed({ metrics }: EnforcementFeedProps) {
  const enforcementEvents = useMemo(
    () => metrics
      .filter((m) => ENFORCEMENT_METRICS.has(m.metric_name))
      .slice(0, 10),
    [metrics],
  );

  if (enforcementEvents.length === 0) {
    return (
      <div className="px-3 py-2 rounded-lg bg-canvas/80 backdrop-blur-sm border border-edge/40">
        <span className="text-[10px] text-ink-faint">No enforcement events</span>
      </div>
    );
  }

  return (
    <div className="w-64 max-h-[220px] overflow-y-auto rounded-lg bg-canvas/80 backdrop-blur-sm border border-edge/40">
      <div className="px-3 py-1.5 border-b border-edge/30">
        <span className="text-[10px] font-semibold text-ink-secondary tracking-wide uppercase">
          Enforcement
        </span>
      </div>
      <div className="divide-y divide-edge/20">
        {enforcementEvents.map((e) => {
          const display = METRIC_DISPLAY[e.metric_name] ?? { label: e.metric_name, dotColor: "bg-ink-faint" };
          const detail = extractDetail(e);
          return (
            <div key={e.id} className="px-3 py-1.5 flex items-start gap-2">
              <span className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${display.dotColor}`} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-medium text-ink-secondary truncate">
                    {display.label}
                  </span>
                  <span className="text-[9px] text-ink-faint shrink-0">
                    {formatTimeAgo(e.created_at)}
                  </span>
                </div>
                {detail && (
                  <div className="text-[9px] text-ink-faint font-mono truncate">{detail}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export { ENFORCEMENT_METRICS };

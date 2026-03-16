import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ProjectAllocation, ProjectRadarCard } from "../lib/types";
import { timeAgo } from "../contentUtils";
import AdminTooltip from "./AdminTooltip";

// ─── Smart narrative templates ───────────────────────

function buildNarrative(p: ProjectRadarCard): string {
  const parts: string[] = [];

  // Activity summary
  const sessionLabel = p.sessionCount === 1 ? "session" : "sessions";
  if (p.momentumDelta > 0 && p.momentum === "up") {
    parts.push(`${p.sessionCount} ${sessionLabel}, up ${p.momentumDelta}% vs last period.`);
  } else if (p.momentum === "steady") {
    parts.push(`${p.sessionCount} ${sessionLabel}, holding steady.`);
  } else if (p.momentum === "cooling") {
    parts.push(`${p.sessionCount} ${sessionLabel}, down ${Math.abs(p.momentumDelta)}%.`);
  } else if (p.momentum === "stalled") {
    parts.push(`No recent sessions.`);
  } else {
    parts.push(`${p.sessionCount} ${sessionLabel} this period.`);
  }

  // Decisions
  if (p.decisionCount > 0) {
    parts.push(`${p.decisionCount} decision${p.decisionCount !== 1 ? "s" : ""} made.`);
  }

  // Tasks
  if (p.taskCompletedCount > 0) {
    parts.push(`${p.taskCompletedCount} task${p.taskCompletedCount !== 1 ? "s" : ""} completed.`);
  }

  // Focus area from most recent decision
  if (p.recentDecisions.length > 0) {
    const latest = p.recentDecisions[0].content;
    const topic = latest.length > 80 ? latest.slice(0, 77) + "..." : latest;
    parts.push(`Latest: ${topic}`);
  }

  return parts.join(" ");
}

function inactiveLabel(p: ProjectRadarCard): string {
  if (p.daysSinceLastSession > 30) return `Last active ${p.daysSinceLastSession} days ago`;
  return `Last active ${timeAgo(p.lastActive)}`;
}

// ─── Momentum indicator ──────────────────────────────

function MomentumBadge({ momentum }: { momentum: ProjectRadarCard["momentum"] }) {
  const config = {
    up: { icon: "\u2191", label: "momentum", cls: "text-type-lesson" },
    steady: { icon: "\u2192", label: "steady", cls: "text-ink-tertiary" },
    cooling: { icon: "\u2193", label: "cooling", cls: "text-type-reminder" },
    stalled: { icon: "\u26A0", label: "stalled", cls: "text-type-error" },
  }[momentum];

  return (
    <span className={`text-[14px] font-mono ${config.cls}`}>
      {config.icon} {config.label}
    </span>
  );
}

function momentumDot(momentum: ProjectRadarCard["momentum"]): string {
  if (momentum === "up") return "bg-type-lesson shadow-[0_0_6px_rgba(94,201,160,0.4)]";
  if (momentum === "steady") return "bg-ink-tertiary";
  if (momentum === "cooling") return "bg-type-reminder shadow-[0_0_6px_rgba(245,158,11,0.4)]";
  return "bg-type-error";
}

// ─── Attention Allocation Bar ─────────────────────────

const ALLOC_COLORS = [
  "var(--color-gold)",
  "var(--color-type-lesson)",
  "var(--color-type-error)",
  "var(--color-ink-tertiary)",
  "var(--color-type-decision)",
];

function AttentionBar({ allocation }: { allocation: ProjectAllocation[] }) {
  if (allocation.length === 0) return null;

  // Only show projects with >0% allocation
  const visible = allocation.filter((a) => a.percent > 0);
  if (visible.length === 0) return null;

  return (
    <div className="mb-5">
      <p className="admin-section-label mb-3">Attention</p>
      <div className="flex rounded-lg overflow-hidden h-6">
        {visible.map((a, i) => (
          <div
            key={a.displayName}
            className="relative group"
            style={{
              width: `${Math.max(a.percent, 2)}%`,
              backgroundColor: ALLOC_COLORS[i % ALLOC_COLORS.length],
              opacity: 0.7 + (a.percent / 100) * 0.3,
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              {a.percent >= 15 && (
                <span className="text-[11px] font-mono text-surface font-medium truncate px-1">
                  {a.displayName} {a.percent}%
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Velocity Sparklines ──────────────────────────────

function VelocityChart({ series }: { series: ProjectRadarCard["velocitySeries"] }) {
  if (!series || series.length === 0) return null;

  return (
    <div>
      <div className="text-[14px] text-ink-tertiary uppercase tracking-wider mb-2">
        Velocity (4 weeks)
      </div>
      <ResponsiveContainer width="100%" height={60}>
        <BarChart data={series} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="week"
            tick={{ fill: "var(--color-ink-tertiary)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<AdminTooltip />} cursor={false} />
          <Bar dataKey="sessions" name="sessions" fill="var(--color-gold)" radius={[2, 2, 0, 0]} barSize={12} fillOpacity={0.7} />
          <Bar dataKey="decisions" name="decisions" fill="var(--color-type-lesson)" radius={[2, 2, 0, 0]} barSize={12} fillOpacity={0.7} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Blockers ─────────────────────────────────────────

function BlockerList({ blockers }: { blockers: ProjectRadarCard["blockers"] }) {
  if (blockers.length === 0) return null;

  return (
    <div>
      <div className="text-[14px] text-ink-tertiary uppercase tracking-wider mb-2">
        Blockers ({blockers.length})
      </div>
      <div className="space-y-1.5">
        {blockers.map((b, i) => (
          <div key={i} className="flex items-start gap-2 text-[16px]">
            <span className="w-1.5 h-1.5 rounded-full bg-type-error shrink-0 mt-2" />
            <span className="text-ink-secondary flex-1 leading-relaxed">
              {b.content.length > 200 ? b.content.slice(0, 197) + "..." : b.content}
            </span>
            <span className="text-ink-tertiary text-[14px] tabular-nums shrink-0">
              {timeAgo(b.createdAt)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Decisions List ───────────────────────────────────

function DecisionList({ decisions }: { decisions: ProjectRadarCard["recentDecisions"] }) {
  if (decisions.length === 0) return null;

  return (
    <div>
      <div className="text-[14px] text-ink-tertiary uppercase tracking-wider mb-2">
        Decisions ({decisions.length})
      </div>
      <div className="max-h-[200px] overflow-y-auto space-y-1.5 pr-1">
        {decisions.map((d, i) => (
          <div key={i} className="flex items-start gap-2 text-[16px]">
            <span className="w-1.5 h-1.5 rounded-full bg-gold shrink-0 mt-2" />
            <span className="text-ink-secondary flex-1 leading-relaxed">
              {d.content.length > 200 ? d.content.slice(0, 197) + "..." : d.content}
            </span>
            <span className="text-ink-tertiary text-[14px] tabular-nums shrink-0">
              {timeAgo(d.createdAt)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Active Project Card ─────────────────────────────

function ActiveProjectCard({
  project,
  isExpanded,
  onToggle,
}: {
  project: ProjectRadarCard;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const narrative = buildNarrative(project);

  return (
    <div className="py-3 border-b border-edge last:border-b-0">
      {/* Header row */}
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`w-2.5 h-2.5 rounded-full shrink-0 ${momentumDot(project.momentum)}`}
          aria-hidden="true"
        />
        <span className="text-[16px] font-medium text-ink">{project.displayName}</span>
        <span className="ml-auto">
          <MomentumBadge momentum={project.momentum} />
        </span>
      </div>

      {/* Narrative summary */}
      <div className="ml-[18px] mb-1.5">
        <p className="text-[16px] text-ink-secondary leading-relaxed">
          {narrative}
        </p>
      </div>

      {/* Blockers inline for active projects */}
      {project.blockers.length > 0 && (
        <div className="ml-[18px] mb-1.5">
          <span className="text-[14px] text-type-error">
            {project.blockers.length} blocker{project.blockers.length !== 1 ? "s" : ""}
          </span>
        </div>
      )}

      {/* Expand toggle */}
      <div className="flex items-center ml-[18px]">
        <button
          onClick={onToggle}
          className="ml-auto flex items-center gap-1.5 text-[14px] text-ink-tertiary hover:text-ink-secondary transition-colors touch-manipulation px-2 py-1 -mr-2 min-h-[44px]"
        >
          <span>{isExpanded ? "Collapse" : "Details"}</span>
          <svg
            width={14}
            height={14}
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
          >
            <path d="M3 4.5L6 7.5L9 4.5" />
          </svg>
        </button>
      </div>

      {/* Expanded detail */}
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: isExpanded ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="ml-[18px] pt-3 space-y-4">
            <DecisionList decisions={project.recentDecisions} />
            <BlockerList blockers={project.blockers} />
            <VelocityChart series={project.velocitySeries} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Inactive Projects Section ───────────────────────

function InactiveSection({ projects }: { projects: ProjectRadarCard[] }) {
  const [showInactive, setShowInactive] = useState(false);

  if (projects.length === 0) return null;

  return (
    <div className="mt-4 pt-3 border-t border-edge">
      <button
        onClick={() => setShowInactive(!showInactive)}
        className="flex items-center gap-2 text-[14px] text-ink-tertiary hover:text-ink-secondary transition-colors touch-manipulation w-full py-1 min-h-[44px]"
      >
        <svg
          width={12}
          height={12}
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform duration-200 ${showInactive ? "rotate-180" : ""}`}
        >
          <path d="M3 4.5L6 7.5L9 4.5" />
        </svg>
        <span>Inactive projects ({projects.length})</span>
      </button>

      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: showInactive ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="mt-2 space-y-1">
            {projects.map((p) => (
              <div
                key={p.displayName}
                className="flex items-center gap-2 py-1.5 text-[14px] text-ink-tertiary"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-edge shrink-0" />
                <span className="font-medium">{p.displayName}</span>
                <span className="ml-auto tabular-nums">{inactiveLabel(p)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────

export default function ProjectRadar({
  allocation,
  projects,
  unscopedCount,
}: {
  allocation: ProjectAllocation[];
  projects: ProjectRadarCard[];
  unscopedCount: number;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const activeProjects = projects.filter((p) => p.isActive);
  const inactiveProjects = projects.filter((p) => !p.isActive);

  // Only show allocation for active projects
  const activeAllocation = allocation.filter((a) =>
    activeProjects.some((p) => p.displayName === a.displayName),
  );

  function toggle(name: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  if (projects.length === 0 && unscopedCount === 0) {
    return (
      <div className="p-4 admin-card">
        <h4 className="admin-section-label mb-3">Project Radar</h4>
        <p className="text-[16px] text-ink-tertiary">
          No project activity yet. Projects appear once coordination is enabled.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 admin-card">
      <h4 className="admin-section-label mb-4">Project Radar</h4>

      <AttentionBar allocation={activeAllocation} />

      {activeProjects.length === 0 ? (
        <p className="text-[16px] text-ink-tertiary py-3">
          No active projects in the last 7 days.
        </p>
      ) : (
        <div className="space-y-1">
          {activeProjects.map((project) => (
            <ActiveProjectCard
              key={project.displayName}
              project={project}
              isExpanded={expanded.has(project.displayName)}
              onToggle={() => toggle(project.displayName)}
            />
          ))}
        </div>
      )}

      <InactiveSection projects={inactiveProjects} />

      {unscopedCount > 0 && (
        <div className="text-[16px] text-ink-tertiary mt-3">
          {unscopedCount} unscoped memor{unscopedCount !== 1 ? "ies" : "y"}
        </div>
      )}
    </div>
  );
}

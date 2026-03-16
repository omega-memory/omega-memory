import React, { useCallback, useEffect, useState } from "react";
import type { WorkflowChangesData } from "./lib/types";

// ─── Helpers ──────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="admin-section-label mb-3">{children}</p>;
}

function directionArrow(dir: "up" | "down" | "flat"): string {
  if (dir === "up") return "\u25B2";
  if (dir === "down") return "\u25BC";
  return "\u2014";
}

function directionColor(dir: "up" | "down" | "flat", label: string): string {
  const downIsGood = ["Commit message size", "Context-switching"].includes(label);
  if (dir === "flat") return "text-ink-tertiary";
  if (dir === "up") return downIsGood ? "text-type-error" : "text-type-lesson";
  return downIsGood ? "text-type-lesson" : "text-type-error";
}

// ─── ChangeList ───────────────────────────────────────

function ChangeList({ changes }: { changes: WorkflowChangesData["changes"] }) {
  if (changes.length === 0) {
    return (
      <p className="text-[16px] text-ink-secondary leading-relaxed">
        No notable changes this week.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {changes.map((c) => (
        <div key={c.label} className="flex items-start gap-2.5">
          <span className={`text-[15px] font-mono shrink-0 mt-0.5 ${directionColor(c.direction, c.label)}`}>
            {directionArrow(c.direction)}
          </span>
          <div className="flex-1">
            <span className="text-[15px] text-ink-primary">
              {c.label} {c.direction === "flat" ? "stable" : `${c.direction} ${c.magnitude}%`}
            </span>
            {c.context && (
              <span className="text-[14px] text-ink-tertiary ml-1.5">
                ({c.context})
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── CoachingCards ─────────────────────────────────────

function CoachingCards({ coaching }: { coaching: WorkflowChangesData["coaching"] }) {
  if (coaching.length === 0) return null;

  return (
    <div className="pt-5 border-t border-edge">
      <SectionLabel>Coaching</SectionLabel>
      <div className="space-y-3">
        {coaching.map((c) => (
          <div
            key={c.id}
            className="rounded-lg bg-surface-elevated/50 border border-edge px-3.5 py-3 space-y-1"
          >
            <div className="flex items-start gap-2.5">
              <span className="inline-block mt-0.5 px-2 py-0.5 text-[14px] font-mono rounded bg-gold/[0.08] text-gold border border-gold/20 shrink-0">
                {c.category}
              </span>
              <span className="text-[16px] text-ink-primary leading-relaxed flex-1">
                {c.text}
              </span>
              {c.trend !== "flat" && (
                <span className={`text-[13px] font-mono shrink-0 mt-0.5 ${
                  c.trend === "improving" ? "text-type-lesson" : "text-type-error"
                }`}>
                  {c.trend === "improving" ? "Improving" : "Getting worse"}
                </span>
              )}
            </div>
            <p className="text-[14px] text-ink-secondary leading-relaxed">
              {c.impact}
            </p>
            {c.evidence && (
              <p className="text-[14px] text-ink-tertiary leading-relaxed">
                {c.evidence}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Strengths ────────────────────────────────────────

function Strengths({ strengths }: { strengths: WorkflowChangesData["strengths"] }) {
  const [expanded, setExpanded] = useState(false);

  if (strengths.length === 0) return null;

  return (
    <div className="pt-5 border-t border-edge">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-ink-tertiary hover:text-ink-secondary transition-colors touch-manipulation"
      >
        <SectionLabel>
          {strengths.length} strength{strengths.length !== 1 ? "s" : ""}
        </SectionLabel>
        <svg
          width={14}
          height={14}
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform duration-200 mb-3 ${expanded ? "rotate-180" : ""}`}
        >
          <path d="M3 4.5L6 7.5L9 4.5" />
        </svg>
      </button>
      {expanded && (
        <div className="space-y-2">
          {strengths.map((wp) => (
            <div key={wp.dimension} className="flex items-start gap-2.5">
              <svg
                className="w-3.5 h-3.5 text-gold shrink-0 mt-[3px]"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M3 8.5l3.5 3.5L13 5" />
              </svg>
              <span className="text-[15px] text-ink-primary leading-relaxed flex-1">
                {wp.description}
              </span>
              <span className="text-[14px] font-mono text-ink-tertiary shrink-0 mt-0.5">
                {wp.dimensionLabel}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────

export default function WorkflowChanges() {
  const [data, setData] = useState<WorkflowChangesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/pattern-insights");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (loading) {
    return (
      <div className="p-4 admin-card">
        <h4 className="admin-section-label">Workflow Changes</h4>
        <div className="mt-4 flex items-center gap-2 text-[16px] text-ink-faint">
          <span className="inline-block w-4 h-4 border-2 border-gold/40 border-t-gold rounded-full animate-spin" />
          Loading...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 admin-card">
        <h4 className="admin-section-label">Workflow Changes</h4>
        <p className="mt-4 text-[16px] text-ink-faint">{error}</p>
      </div>
    );
  }

  if (!data?.hasData) {
    const noSessions = data?.sessionCount === 0;
    return (
      <div className="p-4 admin-card">
        <h4 className="admin-section-label mb-5">Workflow Changes</h4>
        {noSessions ? (
          <>
            <p className="text-[18px] text-ink-primary leading-relaxed">
              No coordination sessions found.
            </p>
            <p className="text-[16px] text-ink-secondary leading-relaxed mt-1">
              Install the OMEGA hooks to see work patterns.
            </p>
          </>
        ) : (
          <>
            <p className="text-[18px] text-ink-primary leading-relaxed">
              OMEGA is still learning your work patterns.
            </p>
            <p className="text-[16px] text-ink-secondary leading-relaxed mt-1">
              Changes will appear after a few sessions with coordination enabled.
            </p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="p-4 admin-card space-y-0">
      <div className="space-y-0">
        <h4 className="admin-section-label mb-3">This Week</h4>
        <ChangeList changes={data.changes} />
        <CoachingCards coaching={data.coaching} />
        <Strengths strengths={data.strengths} />
      </div>
    </div>
  );
}

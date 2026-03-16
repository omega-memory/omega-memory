import React, { useState, useEffect, useCallback, Suspense, lazy } from "react";
import type { ProjectOverview, ArchitectureResponse } from "./types";
import { HEALTH_THEME, computeMomentum, MOMENTUM_THEME, getSparkColor, relativeTime, humanize } from "./utils";
import Sparkline from "./Sparkline";
import { StatCards, BlockersSection, NextStepsSection, DecisionsSection } from "./ProjectDetailSections";
import ActivityTimeline from "./ActivityTimeline";

const ArchitectureDiagram = lazy(() => import("./ArchitectureDiagram"));
const SessionHeatmap = lazy(() => import("./SessionHeatmap"));

type FullViewTab = "overview" | "tasks" | "activity" | "architecture";

interface ProjectFullViewProps {
  project: ProjectOverview;
  onBack: () => void;
}

const TABS: { key: FullViewTab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "tasks", label: "Tasks" },
  { key: "activity", label: "Activity" },
  { key: "architecture", label: "Architecture" },
];

export default function ProjectFullView({ project, onBack }: ProjectFullViewProps) {
  const [activeTab, setActiveTab] = useState<FullViewTab>("overview");
  const [archData, setArchData] = useState<ArchitectureResponse | null>(null);
  const [archLoading, setArchLoading] = useState(false);
  const [archError, setArchError] = useState<string | null>(null);

  const theme = HEALTH_THEME[project.health];
  const momentum = computeMomentum(project);
  const mTheme = MOMENTUM_THEME[momentum];
  const sparkColor = getSparkColor(momentum);
  const handoff = project.latestHandoff;

  // Fetch architecture data on tab switch
  useEffect(() => {
    if (activeTab !== "architecture" || archData) return;
    setArchLoading(true);
    fetch(`/api/admin/projects/${project.id}/architecture`)
      .then((r) => r.ok ? r.json() : Promise.reject("Failed to load"))
      .then((data) => setArchData(data))
      .catch((e) => setArchError(String(e)))
      .finally(() => setArchLoading(false));
  }, [activeTab, project.id, archData]);

  // Listen for popstate (browser back button)
  const stableOnBack = useCallback(() => onBack(), [onBack]);
  useEffect(() => {
    window.addEventListener("popstate", stableOnBack);
    return () => window.removeEventListener("popstate", stableOnBack);
  }, [stableOnBack]);

  // Build task lists
  const tasks = project.tasks;
  const totalTasks = tasks.pending + tasks.inProgress + tasks.completed + tasks.blocked + tasks.failed;
  const remainingItems = tasks.items
    .filter((t) => ["blocked", "in_progress", "pending"].includes(t.status))
    .sort((a, b) => {
      const order: Record<string, number> = { blocked: 0, in_progress: 1, pending: 2 };
      return (order[a.status] ?? 3) - (order[b.status] ?? 3);
    });
  const completedItems = tasks.items.filter((t) => t.status === "completed");

  return (
    <div className="h-full overflow-y-auto">
      {/* Header */}
      <div className="px-8 pt-6 pb-4 border-b border-edge-subtle">
        <div className="flex items-center justify-between">
          <button onClick={onBack} className="text-[14px] text-ink-faint hover:text-ink transition-colors">
            &larr; Back to Projects
          </button>
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${theme.dot}`} />
            <span className="text-[13px] text-ink-faint font-mono">{theme.label}</span>
            <span className={`text-[13px] font-mono ${mTheme.cls}`}>{mTheme.icon} {mTheme.label}</span>
          </div>
        </div>
        <h1 className="text-[28px] font-light text-ink tracking-tight mt-3">{project.name}</h1>
        <p className="text-[14px] text-ink-faint mt-1">
          Last active {relativeTime(project.lastActive)}
          <span className="mx-1.5 text-ink-faint/30">&middot;</span>
          {project.sessionCount30d} sessions / 30d
        </p>

        {/* Tab bar */}
        <div className="flex gap-1 mt-5">
          {TABS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-4 py-2 text-[14px] rounded-t-lg transition-colors ${
                activeTab === key
                  ? "bg-surface-elevated text-ink border-b-2 border-gold"
                  : "text-ink-faint hover:text-ink-secondary"
              }`}
            >
              {label}
              {key === "tasks" && totalTasks > 0 && (
                <span className="ml-1.5 text-[12px] font-mono text-ink-faint">{totalTasks}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="px-8 py-6 space-y-6">
        {activeTab === "overview" && (
          <>
            <StatCards project={project} />
            <Sparkline data={project.sparkline} color={sparkColor} width={700} height={48} />
            {handoff?.blockedItems && <BlockersSection blockedItems={handoff.blockedItems} />}
            {handoff?.nextSteps && <NextStepsSection nextSteps={handoff.nextSteps} />}
            <DecisionsSection decisions={project.decisions} limit={5} />
          </>
        )}

        {activeTab === "tasks" && (
          totalTasks > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div>
                <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
                  Remaining ({remainingItems.length})
                </span>
                <div className="mt-3 space-y-2">
                  {remainingItems.map((t) => (
                    <div key={t.id} className="flex items-start gap-2.5">
                      <span className={`mt-0.5 shrink-0 ${
                        t.status === "blocked" ? "text-type-error" :
                        t.status === "in_progress" ? "text-type-reminder" :
                        "text-ink-faint"
                      }`}>
                        {t.status === "blocked" ? "!" : t.status === "in_progress" ? "\u25B6" : "\u25CB"}
                      </span>
                      <span className={`text-[15px] leading-snug ${
                        t.status === "blocked" ? "text-type-error" : "text-ink-secondary"
                      }`}>{humanize(t.title)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
                  Completed ({completedItems.length})
                </span>
                <div className="mt-3 space-y-2">
                  {completedItems.map((t) => (
                    <div key={t.id} className="flex items-start gap-2.5">
                      <span className="text-type-lesson mt-0.5 shrink-0">&#10003;</span>
                      <span className="text-[15px] text-ink-secondary leading-snug">{humanize(t.title)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-[15px] text-ink-faint">No tasks tracked for this project.</p>
          )
        )}

        {activeTab === "activity" && (
          <ActivityTimeline activity={project.activity} decisions={project.decisions} />
        )}

        {activeTab === "architecture" && (
          <div className="space-y-6">
            {archLoading && (
              <div className="space-y-3">
                <div className="h-[300px] rounded-lg bg-white/[0.03] animate-pulse" />
                <div className="h-[200px] rounded-lg bg-white/[0.03] animate-pulse" />
              </div>
            )}
            {archError && (
              <div className="p-4 rounded-lg bg-type-error/10 border border-type-error/20 text-type-error text-[15px]">
                {archError}
                <button onClick={() => { setArchError(null); setArchData(null); }} className="ml-3 underline">Retry</button>
              </div>
            )}
            {archData && (
              <Suspense fallback={null}>
                {archData.mermaidSyntax && <ArchitectureDiagram mermaidSyntax={archData.mermaidSyntax} />}
                {archData.sessionHeatmap?.length > 0 && <SessionHeatmap data={archData.sessionHeatmap} />}
              </Suspense>
            )}
            {!archLoading && !archError && !archData && (
              <p className="text-[15px] text-ink-faint">No architecture data available.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

import { useState, useEffect, useCallback } from "react";
import type { ProjectOverview, ArchitectureResponse } from "./types";
import { HEALTH_THEME } from "./utils";
import ArchitectureDiagram from "./ArchitectureDiagram";
import SessionHeatmap from "./SessionHeatmap";
import DetailBlockers from "./DetailBlockers";
import DetailNextSteps from "./DetailNextSteps";
import DetailTasks from "./DetailTasks";
import DecisionTimeline from "./DecisionTimeline";
import DetailActivity from "./DetailActivity";
import DisclosureSection from "../shared/DisclosureSection";

interface ProjectDetailViewProps {
  project: ProjectOverview;
  onBack: () => void;
}

export default function ProjectDetailView({
  project,
  onBack,
}: ProjectDetailViewProps) {
  const [archData, setArchData] = useState<ArchitectureResponse | null>(null);
  const [archLoading, setArchLoading] = useState(false);
  const [archExpanded, setArchExpanded] = useState(false);
  const [heatmapExpanded, setHeatmapExpanded] = useState(false);

  const theme = HEALTH_THEME[project.health];
  const blockedItems = project.latestHandoff?.blockedItems ?? [];
  const nextSteps = project.latestHandoff?.nextSteps ?? [];

  const fetchArch = useCallback(async () => {
    setArchLoading(true);
    try {
      const res = await fetch(`/api/admin/projects/${project.id}/architecture`);
      if (res.ok) {
        const data = await res.json();
        setArchData(data);
      }
    } catch {
      // Silent fail
    } finally {
      setArchLoading(false);
    }
  }, [project.id]);

  // Lazy-load architecture only when expanded
  useEffect(() => {
    if (archExpanded && !archData && !archLoading) {
      fetchArch();
    }
  }, [archExpanded, archData, archLoading, fetchArch]);

  // Reset state when project changes
  useEffect(() => {
    setArchData(null);
    setArchExpanded(false);
    setHeatmapExpanded(false);
  }, [project.id]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onBack();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onBack]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-white/[0.06] flex-shrink-0">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/70 transition-colors cursor-pointer"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M8.5 3L4.5 7l4 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Back
        </button>
        <div className="h-4 w-px bg-white/[0.08]" />
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${theme.badge}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${theme.dot}`} />
          {theme.label}
        </span>
        <h1 className="text-sm font-semibold text-white/90">{project.name}</h1>
        {project.hasActiveSession && (
          <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-green-400/10 text-green-400/70">
            agent working
          </span>
        )}
      </div>

      {/* Full-width stacked sections */}
      <div className="flex-1 min-h-0 overflow-y-auto p-6 space-y-4">
        {/* Current Status — always visible */}
        <div className="space-y-4">
          <DetailBlockers blockedItems={blockedItems} />
          <DetailNextSteps nextSteps={nextSteps} />
        </div>

        {/* In Progress Tasks */}
        {project.tasks.items.length > 0 && (
          <DetailTasks tasks={project.tasks} />
        )}

        {/* History — collapsed by default */}
        {(project.decisions.length > 0 || project.activity.length > 0) && (
          <DisclosureSection title="History" storageKey="project-history">
            {project.decisions.length > 0 && (
              <div className="mb-4">
                <DecisionTimeline decisions={project.decisions} />
              </div>
            )}
            {project.activity.length > 0 && (
              <DetailActivity activity={project.activity} />
            )}
          </DisclosureSection>
        )}

        {/* Architecture — collapsed, lazy-loaded */}
        <CollapsibleSection
          title="Architecture diagram"
          expanded={archExpanded}
          onToggle={() => setArchExpanded((v) => !v)}
        >
          {archLoading ? (
            <div className="h-[300px] flex items-center justify-center">
              <span className="text-xs text-white/20 animate-pulse">Loading architecture...</span>
            </div>
          ) : archData?.mermaidSyntax ? (
            <ArchitectureDiagram mermaidSyntax={archData.mermaidSyntax} />
          ) : (
            <div className="h-[200px] flex items-center justify-center">
              <span className="text-xs text-white/20">No architecture data available yet</span>
            </div>
          )}
        </CollapsibleSection>

        {/* Session Heatmap — collapsed */}
        <CollapsibleSection
          title="Session heatmap"
          expanded={heatmapExpanded}
          onToggle={() => {
            setHeatmapExpanded((v) => !v);
            // Ensure arch data is loaded for heatmap
            if (!archData && !archLoading) fetchArch();
          }}
        >
          {archLoading ? (
            <div className="h-[200px] flex items-center justify-center">
              <span className="text-xs text-white/20 animate-pulse">Loading...</span>
            </div>
          ) : archData?.sessionHeatmap && archData.sessionHeatmap.length > 0 ? (
            <SessionHeatmap data={archData.sessionHeatmap} />
          ) : (
            <div className="h-[120px] flex items-center justify-center">
              <span className="text-xs text-white/20">No session data available</span>
            </div>
          )}
        </CollapsibleSection>

        {/* Bottom padding for scroll */}
        <div className="h-4" />
      </div>
    </div>
  );
}

function CollapsibleSection({
  title,
  expanded,
  onToggle,
  children,
}: {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02]">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors rounded-xl cursor-pointer"
      >
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          className={`text-white/30 transition-transform ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M3.5 2l3 3-3 3" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
        </svg>
        <span className="text-xs font-medium text-white/40">
          {title}
        </span>
      </button>
      {expanded && <div className="px-4 pb-4">{children}</div>}
    </div>
  );
}

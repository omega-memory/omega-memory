import { useState } from "react";
import type { ProjectOverview } from "./types";
import DetailBlockers from "./DetailBlockers";
import DetailTasks from "./DetailTasks";
import DetailDecisions from "./DetailDecisions";
import DetailActivity from "./DetailActivity";
import DetailFiles from "./DetailFiles";

interface DetailSidebarProps {
  project: ProjectOverview;
  moduleFilter?: string | null;
  onClearFilter?: () => void;
}

type SidebarSection = "tasks" | "decisions" | "activity" | "files";

export default function DetailSidebar({
  project,
  moduleFilter,
  onClearFilter,
}: DetailSidebarProps) {
  const [expandedSection, setExpandedSection] = useState<SidebarSection | null>(null);

  const filteredFileClaims = moduleFilter
    ? project.fileClaims.filter((c) => c.filePath.includes(moduleFilter))
    : project.fileClaims;

  const filteredActivity = moduleFilter
    ? project.activity.filter(
        (a) => a.type !== "session" || a.summary.toLowerCase().includes(moduleFilter.toLowerCase()),
      )
    : project.activity;

  const toggleSection = (section: SidebarSection) => {
    setExpandedSection((prev) => (prev === section ? null : section));
  };

  return (
    <div className="h-full overflow-y-auto space-y-3 pr-1">
      {moduleFilter && (
        <div className="flex items-center gap-2 rounded-lg bg-teal-400/[0.08] border border-teal-400/20 px-3 py-1.5">
          <span className="text-[10px] text-teal-400/70 font-medium truncate">
            Filtered: {moduleFilter}
          </span>
          <button
            onClick={onClearFilter}
            className="ml-auto text-[10px] text-teal-400/50 hover:text-teal-400/80 transition-colors flex-shrink-0"
          >
            Clear
          </button>
        </div>
      )}

      <DetailBlockers blockedItems={project.latestHandoff?.blockedItems ?? []} />

      <SidebarCollapsible
        title="Tasks"
        count={project.tasks.pending + project.tasks.inProgress}
        expanded={expandedSection === "tasks"}
        onToggle={() => toggleSection("tasks")}
      >
        <DetailTasks tasks={project.tasks} />
      </SidebarCollapsible>

      <SidebarCollapsible
        title="Decisions"
        count={project.decisions.length}
        expanded={expandedSection === "decisions"}
        onToggle={() => toggleSection("decisions")}
      >
        <DetailDecisions decisions={project.decisions} />
      </SidebarCollapsible>

      <SidebarCollapsible
        title="Activity"
        count={filteredActivity.length}
        expanded={expandedSection === "activity"}
        onToggle={() => toggleSection("activity")}
      >
        <DetailActivity activity={filteredActivity} />
      </SidebarCollapsible>

      <SidebarCollapsible
        title="Files in Play"
        count={filteredFileClaims.length}
        expanded={expandedSection === "files"}
        onToggle={() => toggleSection("files")}
      >
        <DetailFiles fileClaims={filteredFileClaims} />
      </SidebarCollapsible>

      <div className="h-6" />
    </div>
  );
}

interface SidebarCollapsibleProps {
  title: string;
  count: number;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function SidebarCollapsible({
  title,
  count,
  expanded,
  onToggle,
  children,
}: SidebarCollapsibleProps) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.015]">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-white/[0.02] transition-colors rounded-lg"
      >
        <span className="text-xs font-semibold text-white/50 uppercase tracking-wider">
          {title}
        </span>
        <div className="flex items-center gap-2">
          {count > 0 && (
            <span className="text-[10px] text-white/25 tabular-nums">{count}</span>
          )}
          <svg
            width="10"
            height="10"
            viewBox="0 0 10 10"
            className={`text-white/20 transition-transform ${expanded ? "rotate-180" : ""}`}
          >
            <path d="M2 3.5l3 3 3-3" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          </svg>
        </div>
      </button>
      {expanded && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}

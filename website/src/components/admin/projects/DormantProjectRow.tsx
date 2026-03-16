import { useState } from "react";
import type { ProjectOverview } from "./types";
import { relativeTime, HEALTH_THEME } from "./utils";

interface DormantProjectRowProps {
  projects: ProjectOverview[];
  onSelectProject: (projectId: string) => void;
  defaultExpanded?: boolean;
}

export default function DormantProjectRow({
  projects,
  onSelectProject,
  defaultExpanded = false,
}: DormantProjectRowProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  if (projects.length === 0) return null;

  return (
    <div className="mt-4">
      {/* Collapsible header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 w-full text-left group py-2"
      >
        <span className="text-[10px] font-medium tracking-wider text-white/25 uppercase">
          Dormant ({projects.length})
        </span>
        <span className="flex-1 h-px bg-white/[0.06]" />
        {!expanded && (
          <span className="text-[10px] text-white/20 truncate max-w-[50%]">
            {projects.map((p) => p.name).join(" \u00b7 ")}
          </span>
        )}
        <span className="text-[10px] text-white/20 group-hover:text-white/40 transition-colors flex-shrink-0">
          {expanded ? "\u25B4" : "\u25BE"}
        </span>
      </button>

      {/* Expanded list */}
      {expanded && (
        <div className="flex flex-col gap-px rounded-lg border border-white/[0.06] bg-[var(--color-surface)] overflow-hidden">
          {projects.map((project) => {
            const theme = HEALTH_THEME[project.health];
            return (
              <button
                key={project.id}
                onClick={() => onSelectProject(project.id)}
                className="flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/[0.03] transition-colors"
              >
                <span className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${theme.dot}`} />
                <span className="text-sm text-white/60 truncate min-w-0">
                  {project.name}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/[0.04] text-white/20 flex-shrink-0">
                  {project.category}
                </span>
                <span className="ml-auto text-[10px] text-white/20 flex-shrink-0">
                  {relativeTime(project.lastActive)}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

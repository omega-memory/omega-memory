import { useMemo } from "react";
import type { ProjectOverview, BoardColumn } from "./types";
import { getProjectColumn } from "./utils";
import ProjectBoardCard from "./ProjectBoardCard";

interface ProjectBoardProps {
  projects: ProjectOverview[];
  searchQuery: string;
  onSelectProject: (projectId: string) => void;
}

const COLUMNS: { key: BoardColumn; label: string; accent: string; accentLine: string }[] = [
  { key: "blocked",    label: "Blocked",    accent: "text-type-error",    accentLine: "bg-type-error/60" },
  { key: "active-now", label: "Active Now", accent: "text-type-lesson",   accentLine: "bg-type-lesson/60" },
  { key: "this-week",  label: "This Week",  accent: "text-type-reminder", accentLine: "bg-type-reminder/60" },
  { key: "steady",     label: "Steady",     accent: "text-ink-faint",     accentLine: "bg-ink-faint/30" },
  { key: "parked",     label: "Parked",     accent: "text-ink-faint/50",  accentLine: "bg-ink-faint/15" },
];

export default function ProjectBoard({ projects, searchQuery, onSelectProject }: ProjectBoardProps) {
  const query = searchQuery.toLowerCase().trim();

  const filtered = useMemo(() => {
    if (!query) return projects;
    return projects.filter((p) =>
      p.name.toLowerCase().includes(query) ||
      p.category.toLowerCase().includes(query) ||
      p.health.includes(query) ||
      (p.summary?.toLowerCase().includes(query) ?? false)
    );
  }, [projects, query]);

  const columns = useMemo(() => {
    const grouped: Record<BoardColumn, ProjectOverview[]> = {
      "blocked": [], "active-now": [], "this-week": [], "steady": [], "parked": [],
    };
    for (const p of filtered) {
      grouped[getProjectColumn(p)].push(p);
    }
    // Sort each column by lastActive descending, nulls last
    for (const col of Object.values(grouped)) {
      col.sort((a, b) => {
        const aTime = a.lastActive ? new Date(a.lastActive).getTime() : 0;
        const bTime = b.lastActive ? new Date(b.lastActive).getTime() : 0;
        return bTime - aTime;
      });
    }
    return grouped;
  }, [filtered]);

  const totalFiltered = filtered.length;

  // All columns empty
  if (totalFiltered === 0) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-[15px] text-ink-faint">
          {query ? "No projects match your filter." : "No projects registered. Projects appear automatically from coordination sessions."}
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-5 overflow-x-auto pb-4 min-h-[400px]">
      {COLUMNS.map(({ key, label, accent, accentLine }) => {
        const cards = columns[key];
        const isEmpty = cards.length === 0;

        return (
          <div
            key={key}
            className={`flex flex-col shrink-0 ${isEmpty ? "w-[120px]" : "w-[280px]"} transition-all`}
          >
            {/* Column header */}
            <div className="mb-3">
              <span className={`text-[12px] font-mono uppercase tracking-wider ${accent}`}>
                {label} ({cards.length})
              </span>
              <div className={`h-[2px] mt-1.5 rounded-full ${accentLine}`} />
            </div>

            {/* Cards */}
            {!isEmpty && (
              <div className="flex flex-col gap-4">
                {cards.map((p) => (
                  <ProjectBoardCard
                    key={p.id}
                    project={p}
                    onClick={() => onSelectProject(p.id)}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

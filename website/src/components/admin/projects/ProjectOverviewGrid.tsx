import { useMemo } from "react";
import type { ProjectOverview } from "./types";
import { isProjectActive } from "./utils";
import ProjectOverviewCard from "./ProjectOverviewCard";
import DormantProjectRow from "./DormantProjectRow";

interface ProjectOverviewGridProps {
  projects: ProjectOverview[];
  searchQuery: string;
  onSelectProject: (projectId: string) => void;
}

export default function ProjectOverviewGrid({
  projects,
  searchQuery,
  onSelectProject,
}: ProjectOverviewGridProps) {
  const query = searchQuery.toLowerCase().trim();
  const filtered = useMemo(
    () =>
      query
        ? projects.filter(
            (p) =>
              p.name.toLowerCase().includes(query) ||
              p.category.toLowerCase().includes(query) ||
              p.health.includes(query),
          )
        : projects,
    [projects, query],
  );

  const { active, dormant } = useMemo(() => {
    const active: ProjectOverview[] = [];
    const dormant: ProjectOverview[] = [];
    for (const p of filtered) {
      if (isProjectActive(p)) {
        active.push(p);
      } else {
        dormant.push(p);
      }
    }
    return { active, dormant };
  }, [filtered]);

  if (filtered.length === 0) {
    return (
      <div className="flex items-center justify-center py-16">
        <p className="text-sm text-white/30">
          {query ? "No projects match your search." : "No projects found."}
        </p>
      </div>
    );
  }

  // Auto-expand dormant section when search matches only dormant projects
  const autoExpandDormant = query.length > 0 && active.length === 0 && dormant.length > 0;

  return (
    <div>
      {active.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(380px,1fr))] gap-3">
          {active.map((project) => (
            <ProjectOverviewCard
              key={project.id}
              project={project}
              onClick={() => onSelectProject(project.id)}
            />
          ))}
        </div>
      )}

      <DormantProjectRow
        projects={dormant}
        onSelectProject={onSelectProject}
        defaultExpanded={autoExpandDormant}
      />
    </div>
  );
}

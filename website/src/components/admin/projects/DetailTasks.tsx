import { useState } from "react";
import type { ProjectOverview } from "./types";

type TaskTab = "in_progress" | "pending" | "completed";

interface DetailTasksProps {
  tasks: ProjectOverview["tasks"];
}

const TAB_CONFIG: { key: TaskTab; label: string; filter: string[] }[] = [
  { key: "in_progress", label: "In Progress", filter: ["in_progress"] },
  { key: "pending", label: "Pending", filter: ["pending", "blocked"] },
  { key: "completed", label: "Completed", filter: ["completed", "failed"] },
];

const STATUS_DOT: Record<string, string> = {
  in_progress: "bg-amber-400",
  pending: "bg-white/30",
  blocked: "bg-red-400",
  completed: "bg-green-400",
  failed: "bg-red-400/60",
};

export default function DetailTasks({ tasks }: DetailTasksProps) {
  const [activeTab, setActiveTab] = useState<TaskTab>("in_progress");

  if (tasks.items.length === 0) return null;

  const tabCounts: Record<TaskTab, number> = {
    in_progress: tasks.inProgress,
    pending: tasks.pending + tasks.blocked,
    completed: tasks.completed + tasks.failed,
  };

  const activeConfig = TAB_CONFIG.find((t) => t.key === activeTab)!;
  const filtered = tasks.items.filter((t) =>
    activeConfig.filter.includes(t.status),
  );

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
        Tasks
      </h3>

      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg bg-white/[0.03] p-0.5">
        {TAB_CONFIG.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-white/[0.08] text-white/80"
                : "text-white/30 hover:text-white/50"
            }`}
          >
            {tab.label}
            {tabCounts[tab.key] > 0 && (
              <span className="ml-1 text-white/20">{tabCounts[tab.key]}</span>
            )}
          </button>
        ))}
      </div>

      {/* Task list */}
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {filtered.length === 0 ? (
          <p className="text-xs text-white/20 py-3 text-center">No tasks</p>
        ) : (
          filtered.map((task) => (
            <div
              key={task.id}
              className="flex items-center gap-2 rounded px-2 py-1.5 text-sm"
            >
              <span
                className={`h-2 w-2 rounded-full flex-shrink-0 ${
                  STATUS_DOT[task.status] ?? "bg-white/20"
                }`}
              />
              <span className="flex-1 text-white/60 truncate text-xs">
                {task.title}
              </span>
              {task.progress > 0 && task.progress < 100 && (
                <span className="text-[10px] text-white/25 flex-shrink-0">
                  {task.progress}%
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

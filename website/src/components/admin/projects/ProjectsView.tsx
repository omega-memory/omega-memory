import { useState, useEffect, useCallback, useMemo, Fragment } from "react";
import type { ProjectOverview, AttentionItem, DecisionItem } from "./types";
import { relativeTime, HEALTH_THEME, humanize, humanizeDecision, isProjectActive, isNoiseActivity, computeMomentum, MOMENTUM_THEME, getProjectColumn, getSparkColor } from "./utils";
import { useKeyboardNav } from "../hooks/useKeyboardNav";
import Sparkline from "./Sparkline";
import NeedsAttentionBar from "./NeedsAttentionBar";
import ProjectOverviewGrid from "./ProjectOverviewGrid";
import SummaryStrip from "./SummaryStrip";
import ActivityTimeline from "./ActivityTimeline";
import ProjectBoard from "./ProjectBoard";
import ProjectDrawer from "./ProjectDrawer";
import ProjectFullView from "./ProjectFullView";

// ─── Helpers ────────────────────────────────────────────────

function HealthDot({ health, active }: { health: string; active?: boolean }) {
  const theme = HEALTH_THEME[health as keyof typeof HEALTH_THEME] ?? HEALTH_THEME.stale;
  return (
    <span className="relative flex h-2.5 w-2.5 shrink-0">
      <span className={`h-2.5 w-2.5 rounded-full ${theme.dot}`} />
      {active && <span className="absolute inset-0 animate-ping rounded-full bg-green-400/60" />}
    </span>
  );
}

function ProgressBar({ done, total }: { done: number; total: number }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const color = pct >= 80 ? "bg-type-lesson/60" : pct >= 40 ? "bg-type-reminder/50" : "bg-ink-faint/30";
  return (
    <div className="flex items-center gap-2.5 min-w-[140px]">
      <div className="flex-1 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${Math.max(pct, 2)}%` }} />
      </div>
      <span className="text-[14px] font-mono tabular-nums text-ink-faint w-10 text-right">{pct}%</span>
    </div>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg className={`w-4 h-4 text-ink-tertiary transition-transform duration-150 ${open ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
    </svg>
  );
}

// ─── Hover Preview Tooltip ───────────────────────────────────

function ProjectHoverPreview({ project }: { project: ProjectOverview }) {
  const momentum = computeMomentum(project);
  const mTheme = MOMENTUM_THEME[momentum];
  const sparkColor = getSparkColor(momentum);
  const totalTasks = project.tasks.pending + project.tasks.inProgress + project.tasks.completed + project.tasks.blocked + project.tasks.failed;
  const handoff = project.latestHandoff;

  return (
    <div className="absolute left-0 top-full mt-1 z-50 w-[320px] rounded-lg border border-edge-subtle bg-[var(--color-surface)] shadow-xl p-4 space-y-3 pointer-events-none opacity-0 group-hover/row:opacity-100 transition-opacity duration-150 delay-300">
      {/* Sparkline + momentum */}
      <div className="flex items-center gap-3">
        <Sparkline data={project.sparkline} color={sparkColor} width={120} height={28} />
        <span className={`text-[13px] font-mono ${mTheme.cls}`}>
          {mTheme.icon} {mTheme.label}
        </span>
      </div>

      {/* Quick stats row */}
      <div className="flex items-center gap-4 text-[12px] font-mono text-ink-faint">
        {project.sessionCount30d > 0 && <span>{project.sessionCount30d} sessions</span>}
        {project.commitCount30d > 0 && <span>{project.commitCount30d} commits</span>}
        {totalTasks > 0 && <span>{project.tasks.completed}/{totalTasks} tasks</span>}
      </div>

      {/* Top blocker or next step */}
      {handoff?.blockedItems?.length ? (
        <div className="text-[13px] text-type-error leading-snug">
          Blocked: {humanize(handoff.blockedItems[0])}
        </div>
      ) : handoff?.nextSteps?.[0] ? (
        <div className="text-[13px] text-ink-secondary leading-snug">
          Next: {humanize(handoff.nextSteps[0])}
        </div>
      ) : null}

      {/* Summary */}
      {project.summary && (
        <div className="text-[12px] text-ink-faint/70 leading-relaxed line-clamp-2">
          {project.summary}
        </div>
      )}
    </div>
  );
}

// ─── Detail Panel (list view only) ──────────────────────────

function ProjectDetailPanel({ project }: { project: ProjectOverview }) {
  const tasks = project.tasks;
  const totalTasks = tasks.pending + tasks.inProgress + tasks.completed + tasks.blocked + tasks.failed;
  const handoff = project.latestHandoff;
  const decisions = project.decisions.slice(0, 5);
  const hasTasks = totalTasks > 0;
  const recentActivity = project.activity.slice(0, 8);

  // Build completed items from task items
  const completedItems: string[] = [];
  for (const t of tasks.items.filter((t) => t.status === "completed")) {
    completedItems.push(humanize(t.title));
  }

  // Build remaining items
  const remainingItems: { text: string; status: "pending" | "blocked" | "in_progress" }[] = [];
  for (const t of tasks.items.filter((t) => t.status === "blocked")) {
    remainingItems.push({ text: humanize(t.title), status: "blocked" });
  }
  for (const t of tasks.items.filter((t) => t.status === "in_progress")) {
    remainingItems.push({ text: humanize(t.title), status: "in_progress" });
  }
  for (const t of tasks.items.filter((t) => t.status === "pending")) {
    remainingItems.push({ text: humanize(t.title), status: "pending" });
  }
  if (remainingItems.length === 0 && handoff?.nextSteps) {
    for (const step of handoff.nextSteps) {
      remainingItems.push({ text: humanize(step), status: "pending" });
    }
  }

  return (
    <div className="admin-card mt-1 mx-2 mb-3 p-5">
      {hasTasks ? (
        /* ── Projects with structured tasks: show completed/remaining ── */
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <div>
            <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
              Completed ({tasks.completed})
            </span>
            <div className="mt-2 space-y-1.5">
              {completedItems.slice(0, 8).map((item, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <span className="text-type-lesson mt-0.5 shrink-0">&#10003;</span>
                  <span className="text-[15px] text-ink-secondary leading-snug">{item}</span>
                </div>
              ))}
              {completedItems.length > 8 && (
                <span className="text-[14px] text-ink-faint">+{completedItems.length - 8} more</span>
              )}
            </div>
          </div>
          <div>
            <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
              Remaining ({remainingItems.length})
            </span>
            <div className="mt-2 space-y-1.5">
              {remainingItems.length > 0 ? remainingItems.slice(0, 8).map((item, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <span className={`mt-0.5 shrink-0 ${
                    item.status === "blocked" ? "text-type-error" :
                    item.status === "in_progress" ? "text-type-reminder" :
                    "text-ink-faint"
                  }`}>
                    {item.status === "blocked" ? "!" : item.status === "in_progress" ? "\u25B6" : "\u25CB"}
                  </span>
                  <span className={`text-[15px] leading-snug ${
                    item.status === "blocked" ? "text-type-error" : "text-ink-secondary"
                  }`}>{item.text}</span>
                </div>
              )) : (
                <span className="text-[15px] text-ink-faint">All tasks completed</span>
              )}
            </div>

            {decisions.length > 0 && (
              <div className="mt-4">
                <span className="text-[13px] font-mono text-ink-faint uppercase tracking-wider">
                  Key Decisions
                </span>
                <div className="mt-2 space-y-2">
                  {decisions.map((d: DecisionItem) => (
                    <div key={d.id} className="border-b border-edge-subtle/30 pb-2 last:border-0">
                      <span className="text-[12px] text-ink-faint font-mono">{relativeTime(d.createdAt)}</span>
                      <p className="text-[15px] text-ink-secondary leading-snug mt-0.5">{humanizeDecision(d.decision)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* ── Projects without tasks: show expandable recent work ── */
        <ActivityTimeline
          activity={recentActivity}
          decisions={decisions}
        />
      )}

      {/* Insight Stat Cards */}
      {(project.sessionCount30d > 0 || project.commitCount30d > 0 || hasTasks) && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {project.sessionCount30d > 0 && (
            <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
              <div className="text-[20px] font-light tabular-nums text-ink">{project.sessionCount30d}</div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Sessions</div>
              <div className="text-[11px] text-ink-faint/50">30 days</div>
            </div>
          )}
          {project.commitCount30d > 0 && (
            <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
              <div className="text-[20px] font-light tabular-nums text-ink">{project.commitCount30d}</div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Commits</div>
              <div className="text-[11px] text-ink-faint/50">30 days</div>
            </div>
          )}
          {project.decisionCount30d > 0 && (
            <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
              <div className="text-[20px] font-light tabular-nums text-ink">{project.decisionCount30d}</div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Decisions</div>
              <div className="text-[11px] text-ink-faint/50">30 days</div>
            </div>
          )}
          {hasTasks && (
            <div className="rounded-lg bg-surface-elevated/40 px-3 py-2.5 border border-edge-subtle/30">
              <div className="text-[20px] font-light tabular-nums text-ink">
                {tasks.completed}<span className="text-[14px] text-ink-faint">/{totalTasks}</span>
              </div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">Tasks Done</div>
              {handoff?.blockedItems?.length ? (
                <div className="text-[11px] text-type-error">{handoff.blockedItems.length} blocked</div>
              ) : (
                <div className="text-[11px] text-ink-faint/50">{Math.round((tasks.completed / totalTasks) * 100)}%</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── View Toggle ─────────────────────────────────────────────

type ViewMode = "board" | "cards" | "list";

function ViewToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  const btn = (m: ViewMode, title: string, icon: React.ReactNode) => (
    <button
      onClick={() => onChange(m)}
      className={`px-2.5 py-1.5 transition-colors ${mode === m ? "bg-surface-elevated text-ink" : "text-ink-faint hover:text-ink-secondary"}`}
      title={title}
    >
      {icon}
    </button>
  );

  return (
    <div className="flex items-center rounded-lg border border-edge overflow-hidden">
      {btn("board", "Board view",
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="4" width="3.5" height="10" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="6.25" y="2" width="3.5" height="12" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="11.5" y="6" width="3.5" height="8" rx="1" stroke="currentColor" strokeWidth="1.2" />
        </svg>
      )}
      {btn("cards", "Card view",
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <rect x="1.5" y="1.5" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="9.5" y="1.5" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="1.5" y="9.5" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="9.5" y="9.5" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" />
        </svg>
      )}
      {btn("list", "List view",
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
      )}
    </div>
  );
}

// ─── Main View ──────────────────────────────────────────────

export default function ProjectsView() {
  const [projects, setProjects] = useState<ProjectOverview[]>([]);
  const [needsAttention, setNeedsAttention] = useState<AttentionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("board");

  // Drawer + full-view state
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [fullViewProjectId, setFullViewProjectId] = useState<string | null>(null);

  // List view only
  const [listExpandedId, setListExpandedId] = useState<string | null>(null);
  const [dormantExpanded, setDormantExpanded] = useState(false);

  // Data fetching
  const fetchOverview = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/projects/overview");
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setProjects(data.projects ?? []);
      setNeedsAttention(data.needsAttention ?? []);
    } catch {
      setError("Failed to load projects");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchOverview();
    const interval = setInterval(fetchOverview, 60_000);
    return () => clearInterval(interval);
  }, [fetchOverview]);

  // Deep-link restore on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const projectParam = params.get("project");
    if (projectParam) setFullViewProjectId(projectParam);
  }, []);

  // Derive non-parked projects for SummaryStrip
  const nonParked = useMemo(
    () => projects.filter((p) => getProjectColumn(p) !== "parked"),
    [projects],
  );
  const dormantCount = useMemo(
    () => projects.filter((p) => getProjectColumn(p) === "parked").length,
    [projects],
  );

  // List view filtering/sorting
  const query = searchQuery.toLowerCase();
  const filtered = projects.filter((p) =>
    !query || p.name.toLowerCase().includes(query) || (p.summary?.toLowerCase().includes(query) ?? false)
  );
  const active = filtered.filter((p) => isProjectActive(p)).sort((a, b) => {
    if (a.health === "blocked" && b.health !== "blocked") return -1;
    if (b.health === "blocked" && a.health !== "blocked") return 1;
    const aTime = a.lastActive ? new Date(a.lastActive).getTime() : 0;
    const bTime = b.lastActive ? new Date(b.lastActive).getTime() : 0;
    return bTime - aTime;
  });
  const dormant = filtered.filter((p) => !isProjectActive(p));

  // Combine active + dormant IDs for keyboard navigation
  const allProjectIds = useMemo(
    () => [...active.map((p) => p.id), ...dormant.map((p) => p.id)],
    [active, dormant],
  );

  const { focusedId } = useKeyboardNav({
    dataAttribute: "data-project-id",
    ids: allProjectIds,
    onSelect: (id) => setListExpandedId((prev) => (prev === id ? null : id)),
    enabled: !loading && !selectedProjectId && !fullViewProjectId,
  });

  // Drawer/full-view helpers
  const selectedProject = selectedProjectId ? projects.find((p) => p.id === selectedProjectId) ?? null : null;
  const fullViewProject = fullViewProjectId ? projects.find((p) => p.id === fullViewProjectId) ?? null : null;

  const openFullView = useCallback((projectId: string) => {
    setSelectedProjectId(null);
    setFullViewProjectId(projectId);
    const url = new URL(window.location.href);
    url.searchParams.set("project", projectId);
    window.history.pushState({}, "", url.toString());
  }, []);

  const closeFullView = useCallback(() => {
    setFullViewProjectId(null);
    const url = new URL(window.location.href);
    url.searchParams.delete("project");
    url.searchParams.delete("view");
    window.history.pushState({}, "", url.toString());
  }, []);

  // Loading skeleton
  if (loading) {
    return (
      <div className="p-6 space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-14 rounded-lg bg-white/[0.03] animate-pulse" />
        ))}
      </div>
    );
  }

  // Full-page view takes over
  if (fullViewProject) {
    return <ProjectFullView project={fullViewProject} onBack={closeFullView} />;
  }

  return (
    <div className="h-full overflow-y-auto px-6 sm:px-8 py-6 space-y-5">
      {error && (
        <div className="p-4 rounded-lg bg-type-error/10 border border-type-error/20 text-type-error text-[15px]">
          {error}
        </div>
      )}

      {/* Summary Stat Strip */}
      <SummaryStrip
        projects={nonParked}
        dormantCount={dormantCount}
        attentionCount={needsAttention.filter((a) => a.severity === "red").length}
      />

      {/* Needs Attention Banner — click opens drawer */}
      <NeedsAttentionBar
        items={needsAttention}
        onClickItem={(id) => setSelectedProjectId(id)}
      />

      {/* Header + view toggle + search */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <h1 className="text-[22px] font-light text-ink tracking-tight">Projects</h1>
          <ViewToggle mode={viewMode} onChange={setViewMode} />
        </div>
        <div className="relative w-56">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint/40" width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.2" />
            <path d="M9.5 9.5L13 13" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
          <input
            type="text"
            placeholder="Filter..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-edge bg-surface py-2 pl-9 pr-3 text-[15px] text-ink placeholder:text-ink-faint/40 outline-none focus:border-edge-default transition-colors"
          />
        </div>
      </div>

      {/* Board View */}
      {viewMode === "board" && (
        <ProjectBoard
          projects={projects}
          searchQuery={searchQuery}
          onSelectProject={(id) => setSelectedProjectId(id)}
        />
      )}

      {/* Card View */}
      {viewMode === "cards" && (
        <ProjectOverviewGrid
          projects={projects}
          searchQuery={searchQuery}
          onSelectProject={(id) => setSelectedProjectId(id)}
        />
      )}

      {/* List View */}
      {viewMode === "list" && (
      <div className="admin-card overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-edge-subtle">
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint w-8"></th>
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint">Project</th>
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint hidden sm:table-cell">Progress</th>
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint hidden md:table-cell w-[140px]">Trend</th>
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint hidden md:table-cell">Last Active</th>
              <th className="text-left py-3 px-4 text-[12px] font-mono uppercase tracking-wider text-ink-faint hidden lg:table-cell">Next Step</th>
              <th className="w-8 py-3 px-4"></th>
            </tr>
          </thead>
          <tbody>
            {active.map((p) => {
              const isOpen = listExpandedId === p.id;
              const totalTasks = p.tasks.pending + p.tasks.inProgress + p.tasks.completed + p.tasks.blocked + p.tasks.failed;
              const progressDone = totalTasks > 0 ? p.tasks.completed : p.launchProgress;
              const progressTotal = totalTasks > 0 ? totalTasks : 100;
              const momentum = computeMomentum(p);
              const mTheme = MOMENTUM_THEME[momentum];
              const sparkColor = getSparkColor(momentum);
              // Next step
              const rawNext = p.latestHandoff?.nextSteps?.[0]
                ? humanize(p.latestHandoff.nextSteps[0])
                : p.latestHandoff?.blockedItems?.[0]
                  ? humanize(p.latestHandoff.blockedItems[0])
                  : null;
              const firstCleanActivity = p.activity
                .map((a) => humanize(a.summary))
                .find((s) => s.length >= 10 && s !== "Session" && !isNoiseActivity(s));
              const nextStep = rawNext && rawNext.length > 10
                ? rawNext
                : p.tasks.pending + p.tasks.inProgress > 0
                  ? `${p.tasks.pending + p.tasks.inProgress} tasks in progress`
                  : firstCleanActivity
                    ? firstCleanActivity
                    : "No pending work";
              const nextStepTruncated = nextStep.length > 60 ? nextStep.slice(0, 57) + "..." : nextStep;

              const isFocused = focusedId === p.id;
              // Subtle health-tinted row background
              const healthBg = p.health === "blocked" ? "bg-red-400/[0.03]"
                : p.health === "attention" ? "bg-amber-400/[0.03]"
                : "";

              return (
                <Fragment key={p.id}>
                  <tr
                    data-project-id={p.id}
                    onClick={() => setListExpandedId(isOpen ? null : p.id)}
                    className={`group/row border-b border-edge-subtle/50 cursor-pointer transition-colors ${healthBg} ${
                      isOpen ? "bg-surface-elevated/30" : "hover:bg-surface-elevated/20"
                    } ${isFocused ? "ring-1 ring-gold/40 bg-gold/[0.03]" : ""}`}
                  >
                    <td className="py-3.5 px-4">
                      <HealthDot health={p.health} active={p.hasActiveSession} />
                    </td>
                    <td className="py-3.5 px-4 relative">
                      <div className="text-[16px] font-medium text-ink">{p.name}</div>
                      {p.summary && (
                        <div className="text-[14px] text-ink-faint mt-0.5 truncate max-w-[280px]">
                          {p.category !== "other" ? p.category : ""}
                        </div>
                      )}
                      {!isOpen && <ProjectHoverPreview project={p} />}
                    </td>
                    <td className="py-3.5 px-4 hidden sm:table-cell">
                      <ProgressBar done={progressDone} total={progressTotal} />
                    </td>
                    <td className="py-3.5 px-4 hidden md:table-cell">
                      <div className="flex items-center gap-2">
                        <Sparkline data={p.sparkline} color={sparkColor} width={80} height={24} />
                        <span className={`text-[13px] font-mono ${mTheme.cls}`}>
                          {mTheme.icon}
                        </span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 hidden md:table-cell">
                      <span className="text-[15px] font-mono tabular-nums text-ink-faint">
                        {relativeTime(p.lastActive)}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 hidden lg:table-cell">
                      <span className={`text-[15px] ${
                        p.latestHandoff?.blockedItems?.length ? "text-type-error" : "text-ink-secondary"
                      } truncate block max-w-[280px]`}>
                        {p.latestHandoff?.blockedItems?.length ? "Blocked: " : ""}
                        {nextStepTruncated}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <ChevronIcon open={isOpen} />
                    </td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={7} className="p-0">
                        <ProjectDetailPanel project={p} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}

            {/* Dormant projects — collapsible section */}
            {dormant.length > 0 && (
              <>
                <tr>
                  <td colSpan={7} className="py-3 px-4">
                    <button
                      onClick={() => setDormantExpanded((v) => !v)}
                      className="flex items-center gap-2 text-[13px] font-mono text-ink-faint uppercase tracking-wider hover:text-ink-secondary transition-colors"
                    >
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 12 12"
                        fill="none"
                        className={`transition-transform ${dormantExpanded ? "" : "-rotate-90"}`}
                      >
                        <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      {dormant.length} dormant project{dormant.length !== 1 ? "s" : ""}
                      {!dormantExpanded && (
                        <span className="normal-case tracking-normal font-sans text-ink-faint/50 ml-1">
                          {dormant.slice(0, 4).map((p) => p.name).join(", ")}{dormant.length > 4 ? `, +${dormant.length - 4}` : ""}
                        </span>
                      )}
                    </button>
                  </td>
                </tr>
                {dormantExpanded && dormant.map((p) => {
                  const isOpen = listExpandedId === p.id;
                  const isDormantFocused = focusedId === p.id;
                  const dTotalTasks = p.tasks.pending + p.tasks.inProgress + p.tasks.completed + p.tasks.blocked + p.tasks.failed;
                  const dProgressDone = dTotalTasks > 0 ? p.tasks.completed : p.launchProgress;
                  const dProgressTotal = dTotalTasks > 0 ? dTotalTasks : 100;
                  return (
                    <Fragment key={p.id}>
                      <tr
                        data-project-id={p.id}
                        onClick={() => setListExpandedId(isOpen ? null : p.id)}
                        className={`border-b border-edge-subtle/30 cursor-pointer opacity-50 hover:opacity-70 transition-all ${
                          isOpen ? "bg-surface-elevated/20 !opacity-70" : ""
                        } ${isDormantFocused ? "ring-1 ring-gold/40 !opacity-70 bg-gold/[0.03]" : ""}`}
                      >
                        <td className="py-3 px-4">
                          <HealthDot health={p.health} />
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-[16px] font-medium text-ink">{p.name}</span>
                        </td>
                        <td className="py-3 px-4 hidden sm:table-cell">
                          <ProgressBar done={dProgressDone} total={dProgressTotal} />
                        </td>
                        <td className="py-3 px-4 hidden md:table-cell">
                          <Sparkline data={p.sparkline} color="rgba(255,255,255,0.2)" width={80} height={24} />
                        </td>
                        <td className="py-3 px-4 hidden md:table-cell">
                          <span className="text-[15px] font-mono tabular-nums text-ink-faint">
                            {relativeTime(p.lastActive)}
                          </span>
                        </td>
                        <td className="py-3 px-4 hidden lg:table-cell">
                          <span className="text-[15px] text-ink-faint">No pending work</span>
                        </td>
                        <td className="py-3 px-4">
                          <ChevronIcon open={isOpen} />
                        </td>
                      </tr>
                      {isOpen && (
                        <tr>
                          <td colSpan={7} className="p-0">
                            <ProjectDetailPanel project={p} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </>
            )}
          </tbody>
        </table>
      </div>
      )}

      {/* Project Drawer — always rendered, shown when selectedProjectId is set */}
      <ProjectDrawer
        project={selectedProject}
        onClose={() => setSelectedProjectId(null)}
        onOpenFull={openFullView}
      />
    </div>
  );
}

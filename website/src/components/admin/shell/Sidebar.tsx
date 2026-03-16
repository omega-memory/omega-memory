import { type ReactElement, useEffect, useState } from "react";
import { type Tab, CONTRIBUTOR_TABS, SIDEBAR_GROUPS, type SidebarGroup } from "../lib/types";

interface SidebarProps {
  active: Tab;
  onNavigate: (tab: Tab) => void;
  actionsBadge: number;
  activeAgents?: number;
  recentMemories?: number;
  hasConflicts?: boolean;
  sparkline?: number[];
  jobIssues?: number;
  pendingApprovals?: number;
  researchBadge?: number;
  role?: "owner" | "contributor" | null;
}

const STORAGE_KEY = "omega-sidebar-collapsed";
const GROUPS_STORAGE_KEY = "omega-sidebar-groups";

// ─── Flat icon map ────────────────────────────────────────────

const TAB_ICONS: Record<string, ReactElement> = {
  dashboard: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25a2.25 2.25 0 0 1-2.25-2.25v-2.25Z" />
    </svg>
  ),
  projects: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 0 1 4.5 9.75h15A2.25 2.25 0 0 1 21.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z" />
    </svg>
  ),
  feed: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
    </svg>
  ),
  actions: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 22 22" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M11 1.5L3 5v5.5c0 4.7 3.4 9.1 8 10 4.6-.9 8-5.3 8-10V5l-8-3.5Z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 11l2.5 2.5 4.5-5" />
    </svg>
  ),
  research: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25" />
    </svg>
  ),
  docs: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 22 22" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 4.5c0-.8.7-1.5 1.5-1.5H9c.6 0 1.1.3 1.4.7L11 5l.6-1.3c.3-.4.8-.7 1.4-.7h4.5c.8 0 1.5.7 1.5 1.5v13c0 .8-.7 1.5-1.5 1.5h-5l-1.5-1-1.5 1h-5C3.7 19 3 18.3 3 17.5v-13Z" />
      <path strokeLinecap="round" d="M11 5v14" />
    </svg>
  ),
  growth: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941" />
    </svg>
  ),
  coordination: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
    </svg>
  ),
  diagnostic: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
    </svg>
  ),
  entities: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3H21m-3.75 3H21" />
    </svg>
  ),
  jobs: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 22 22" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 3v2M16 3v2M3 8h16M3 8v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8M8 12h2v2H8v-2Zm4 0h2v2h-2v-2Z" />
    </svg>
  ),
  settings: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  ),
  insights: (
    <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5m.75-9 3-3 2.148 2.148A12.061 12.061 0 0 1 16.5 7.605" />
    </svg>
  ),
};

// Tab labels (reuse from types or define inline)
const TAB_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  projects: "Projects",
  feed: "Feed",
  actions: "Social",
  insights: "Insights",
  research: "Research",
  docs: "Docs",
  growth: "Growth",
  coordination: "Coordination",
  diagnostic: "Diagnostic",
  entities: "Entities",
  jobs: "Jobs",
  settings: "Settings",
};

// Graph links (separate routes, not tabs)
const graphLinks: { href: string; label: string; icon: ReactElement }[] = [
  {
    href: "/admin/memories/graph",
    label: "Memory Graph",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <circle cx="5" cy="6" r="2" />
        <circle cx="19" cy="6" r="2" />
        <circle cx="12" cy="18" r="2" />
        <path strokeLinecap="round" d="M6.5 7.5L11 16.5M17.5 7.5L13 16.5M7 6h10" />
      </svg>
    ),
  },
  {
    href: "/admin/skills/graph",
    label: "Skills Graph",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
      </svg>
    ),
  },
];

function MiniSparkline({ data }: { data: number[] }) {
  if (data.length === 0 || data.every((v) => v === 0)) return null;
  const max = Math.max(...data, 1);
  const h = 20;
  const w = 80;
  const step = w / (data.length - 1 || 1);
  const points = data.map((v, i) => `${i * step},${h - (v / max) * (h - 2)}`).join(" ");
  return (
    <svg width={w} height={h} className="opacity-60" preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-gold"
      />
    </svg>
  );
}

// Shared nav item renderer
function NavItem({
  isActive,
  collapsed,
  icon,
  label,
  badge,
  onClick,
}: {
  isActive: boolean;
  collapsed: boolean;
  icon: ReactElement;
  label: string;
  badge?: ReactElement | null;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={collapsed ? label : undefined}
      className={`sidebar-item relative flex items-center gap-3 rounded-lg transition-all duration-150 w-full ${
        collapsed ? "justify-center px-0 py-2" : "px-3 py-1.5"
      } ${
        isActive
          ? "sidebar-item-active bg-gold/[0.07] text-gold"
          : "text-ink-tertiary hover:text-ink-secondary hover:bg-white/[0.03]"
      }`}
    >
      {isActive && (
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[2.5px] h-5 rounded-r bg-gold shadow-[0_0_12px_rgba(212,168,67,0.5)]" />
      )}
      <span className="shrink-0">{icon}</span>
      {!collapsed && <span className="text-[14px] font-medium">{label}</span>}
      {badge && !collapsed && <span className="ml-auto">{badge}</span>}
      {badge && collapsed && <span className="absolute -top-0.5 -right-0.5">{badge}</span>}
    </button>
  );
}

function CollapsibleGroupLabel({
  label,
  collapsed: sidebarCollapsed,
  isGroupOpen,
  onToggle,
  badge,
}: {
  label: string;
  collapsed: boolean;
  isGroupOpen: boolean;
  onToggle: () => void;
  badge?: number;
}) {
  if (sidebarCollapsed) return <div className="my-2 mx-2 h-px bg-edge-subtle" />;
  return (
    <button
      onClick={onToggle}
      className="w-full flex items-center gap-1 px-3 pt-3 pb-0.5 group/label cursor-pointer"
      aria-expanded={isGroupOpen}
    >
      <svg
        className={`w-3 h-3 text-ink-faint/30 transition-transform duration-150 ${isGroupOpen ? "rotate-90" : ""}`}
        fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
      </svg>
      <span className="text-[11px] uppercase tracking-[0.12em] text-ink-faint/40 font-mono group-hover/label:text-ink-faint/60 transition-colors">
        {label}
      </span>
      {!isGroupOpen && badge != null && badge > 0 && (
        <span className="ml-auto min-w-[18px] h-[18px] text-[10px] font-bold rounded-full flex items-center justify-center px-1.5 bg-ink-faint/10 text-ink-faint">
          {badge}
        </span>
      )}
    </button>
  );
}

export default function Sidebar({ active, onNavigate, actionsBadge, activeAgents = 0, recentMemories = 0, hasConflicts = false, sparkline = [], jobIssues = 0, pendingApprovals = 0, researchBadge = 0, role = "owner" }: SidebarProps) {
  const isOwner = role === "owner";
  const canSee = (id: Tab) => isOwner || CONTRIBUTOR_TABS.has(id);
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(STORAGE_KEY) === "1";
    }
    return false;
  });

  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => {
    if (typeof window !== "undefined") {
      try {
        const stored = JSON.parse(localStorage.getItem(GROUPS_STORAGE_KEY) || "[]");
        return new Set(stored);
      } catch { return new Set(); }
    }
    return new Set();
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, collapsed ? "1" : "0");
  }, [collapsed]);

  useEffect(() => {
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify([...collapsedGroups]));
  }, [collapsedGroups]);

  const toggleGroup = (key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const badgeEl = (bg: string, _text: string, count: number | string) => (
    <span className={`min-w-[18px] h-[18px] text-[10px] font-bold rounded-full flex items-center justify-center px-1.5 ${bg}`}>
      {count}
    </span>
  );

  // Compute badge for a single tab
  const tabBadge = (tabId: Tab): ReactElement | null => {
    if (tabId === "actions" && actionsBadge > 0) {
      return badgeEl("bg-type-error text-white shadow-[0_0_8px_rgba(240,96,96,0.4)]", "error", actionsBadge > 9 ? "9+" : actionsBadge);
    }
    if (tabId === "feed" && recentMemories > 0) {
      return badgeEl("bg-gold/20 text-gold", "gold", recentMemories > 99 ? "99+" : recentMemories);
    }
    if (tabId === "research" && researchBadge > 0) {
      return badgeEl("bg-emerald-500/20 text-emerald-400", "", researchBadge);
    }
    if (tabId === "coordination" && activeAgents > 0) {
      return badgeEl(
        hasConflicts
          ? "bg-type-error/20 text-type-error shadow-[0_0_8px_rgba(240,96,96,0.3)]"
          : "bg-emerald-500/20 text-emerald-400",
        "",
        activeAgents,
      );
    }
    if (tabId === "jobs" && ((jobIssues ?? 0) > 0 || (pendingApprovals ?? 0) > 0)) {
      return (
        <span className="flex items-center gap-1">
          {(jobIssues ?? 0) > 0 && badgeEl("bg-type-error/20 text-type-error shadow-[0_0_8px_rgba(240,96,96,0.3)]", "", jobIssues)}
          {(pendingApprovals ?? 0) > 0 && badgeEl("bg-type-reminder/20 text-type-reminder", "", pendingApprovals)}
        </span>
      );
    }
    return null;
  };

  // Compute aggregate badges per group
  const groupBadge = (group: SidebarGroup): number => {
    let total = 0;
    for (const tab of group.tabs) {
      if (tab === "actions" && actionsBadge > 0) total += actionsBadge;
      if (tab === "feed" && recentMemories > 0) total += recentMemories;
      if (tab === "jobs" && (jobIssues > 0 || pendingApprovals > 0)) total += jobIssues + pendingApprovals;
      if (tab === "coordination" && activeAgents > 0) total += activeAgents;
      if (tab === "research" && researchBadge > 0) total += researchBadge;
    }
    return total;
  };

  return (
    <aside
      className={`sidebar-nav hidden lg:flex flex-col shrink-0 h-screen sticky top-0 z-30 transition-[width] duration-200 ease-out canvas-grain ${
        collapsed ? "w-16" : "w-[232px]"
      }`}
    >
      {/* Branding */}
      <div className={`flex items-center h-[52px] px-5 shrink-0 ${collapsed ? "justify-center px-0" : ""}`}>
        {collapsed ? (
          <span className="font-mono text-[13px] font-semibold tracking-[0.15em] uppercase text-white">
            &Omega;
          </span>
        ) : (
          <>
            <span className="font-mono text-[13px] tracking-[0.15em] uppercase text-white/90 font-medium">OMEGA</span>
            <span className="w-1.5 h-1.5 rounded-full bg-gold mx-2.5 shrink-0 shadow-[0_0_8px_rgba(212,168,67,0.7)]" />
            <span className="text-[12px] text-ink-faint/50 tracking-wide font-mono">Admin</span>
          </>
        )}
      </div>

      {/* Gold divider */}
      <div className="mx-4 h-px bg-gradient-to-r from-transparent via-gold/20 to-transparent" />

      {/* Navigation */}
      <nav className="flex-1 flex flex-col px-2.5 pt-2 gap-px overflow-y-auto scrollbar-hide" role="navigation" aria-label="Main navigation">
        {SIDEBAR_GROUPS.map((group) => {
          const visibleTabs = group.tabs.filter(canSee);
          if (visibleTabs.length === 0) return null;
          const isGroupOpen = !collapsedGroups.has(group.key);
          return (
            <div key={group.key}>
              <CollapsibleGroupLabel
                label={group.label}
                collapsed={collapsed}
                isGroupOpen={isGroupOpen}
                onToggle={() => toggleGroup(group.key)}
                badge={groupBadge(group)}
              />
              {isGroupOpen && visibleTabs.map((tabId) => (
                <NavItem
                  key={tabId}
                  isActive={active === tabId}
                  collapsed={collapsed}
                  icon={TAB_ICONS[tabId]}
                  label={TAB_LABELS[tabId] ?? tabId}
                  onClick={() => onNavigate(tabId)}
                  badge={tabBadge(tabId)}
                />
              ))}
              {/* Graph links after Operate group */}
              {group.key === "operate" && isGroupOpen && graphLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  title={collapsed ? link.label : undefined}
                  className={`sidebar-item relative flex items-center gap-3 rounded-lg transition-all duration-150 ${
                    collapsed ? "justify-center px-0 py-2" : "px-3 py-1.5"
                  } text-ink-tertiary hover:text-ink-secondary hover:bg-white/[0.03]`}
                >
                  <span className="shrink-0">{link.icon}</span>
                  {!collapsed && <span className="text-[14px] font-medium">{link.label}</span>}
                </a>
              ))}
            </div>
          );
        })}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Activity sparkline */}
        {!collapsed && sparkline.length > 0 && !sparkline.every((v) => v === 0) && (
          <div className="px-3 py-2 mt-1">
            <div className="text-[10px] uppercase tracking-[0.12em] text-ink-faint/30 font-mono mb-1.5">
              Activity 24h
            </div>
            <MiniSparkline data={sparkline} />
          </div>
        )}

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-2 px-3 py-3 mb-2 text-ink-faint/30 hover:text-ink-tertiary transition-colors cursor-pointer"
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <svg className={`w-4 h-4 transition-transform duration-200 ${collapsed ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5" />
          </svg>
          {!collapsed && <span className="text-[12px]">Collapse</span>}
        </button>
      </nav>
    </aside>
  );
}

import React, { Component, useState, useCallback, useEffect, useMemo, Suspense, type ReactNode } from "react";
import Sidebar from "./shell/Sidebar";
import type { XAccount } from "./lib/types";
import { TABS, TAB_LABELS, type Tab, getVisibleTabs } from "./lib/types";

const TAB_WIDTH_MAP: Partial<Record<Tab, string>> = {
  dashboard: "max-w-5xl",
  feed: "max-w-5xl",
  actions: "max-w-5xl",
  research: "max-w-5xl",
  docs: "max-w-5xl",
  jobs: "max-w-6xl",
  entities: "max-w-6xl",
  diagnostic: "max-w-6xl",
  growth: "max-w-6xl",
  insights: "max-w-4xl",
  settings: "max-w-4xl",
};
import OmegaMark from "./OmegaMark";
import TopBar from "./shell/TopBar";
import MobileNav from "./shell/MobileNav";
import PasskeyBanner from "./PasskeyBanner";
import { useAmbientStatus } from "./hooks/useAmbientStatus";
import { useCurrentUser } from "./hooks/useCurrentUser";
import { useGenerateAction } from "./hooks/useGenerateAction";
import { useSmartPoll } from "./hooks/useSmartPoll";
import "./admin-theme.css";

const AmbientToast = React.lazy(() => import("./shell/AmbientToast"));
const CommandPalette = React.lazy(() => import("./shell/CommandPalette"));

// ─── Lazy-loaded tab components ─────────────────────────────
const Dashboard = React.lazy(() => import("./Dashboard"));
const Feed = React.lazy(() => import("./Feed"));
const ActionsView = React.lazy(() => import("./ActionsView"));
const Insights = React.lazy(() => import("./Insights"));
const KnowledgeBase = React.lazy(() => import("./KnowledgeBase"));
const Schedules = React.lazy(() => import("./jobs/JobList"));
const Settings = React.lazy(() => import("./Settings"));
const Coordination = React.lazy(() => import("./coordination/CoordinationTab"));
const Entities = React.lazy(() => import("./entities/EntitiesTab"));
const Projects = React.lazy(() => import("./projects"));
const GrowthTab = React.lazy(() => import("./growth/GrowthTab"));
const Research = React.lazy(() => import("./Research"));
const Diagnostic = React.lazy(() => import("./Diagnostic"));


function TabSkeleton() {
  return (
    <div className="px-5 pt-6 space-y-4">
      <div className="h-8 w-48 rounded-lg skeleton" />
      <div className="h-24 w-full rounded-xl skeleton" />
      <div className="h-24 w-full rounded-xl skeleton" />
    </div>
  );
}

// ─── Error boundary for tab isolation ────────────────────────
interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackLabel?: string;
}
interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
  retryKey: number;
}
class TabErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, errorMessage: "", retryKey: 0 };
  }
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, errorMessage: `${error.name}: ${error.message}` };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[TabErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="px-5 pt-10 flex flex-col items-center gap-3 text-center">
          <div className="text-[14px] text-ink-secondary">
            {this.props.fallbackLabel ?? "This tab encountered an error."}
          </div>
          {this.state.errorMessage && (
            <div className="text-[13px] text-type-error/80 font-mono bg-type-error/5 border border-type-error/20 rounded-lg px-4 py-2 max-w-[600px] break-words">
              {this.state.errorMessage}
            </div>
          )}
          <button
            onClick={() => this.setState({ hasError: false, errorMessage: "", retryKey: this.state.retryKey + 1 })}
            className="text-[13px] font-medium px-4 py-2 rounded-lg bg-surface-elevated border border-edge hover:bg-surface-hover transition-colors text-ink-secondary"
          >
            Click to retry
          </button>
        </div>
      );
    }
    return <div key={this.state.retryKey} className="flex-1 min-h-0 flex flex-col [&>*]:flex-1 [&>*]:min-h-0">{this.props.children}</div>;
  }
}

// Backward compat mapping for old URL params
const TAB_ALIASES: Record<string, Tab> = {
  review: "actions",
  chat: "feed",
  kb: "docs",
  schedules: "jobs",
};

function resolveTab(raw: string | null): { tab: Tab; openPalette: boolean } {
  if (!raw) return { tab: "dashboard", openPalette: false };
  if (TABS.includes(raw as Tab)) {
    return { tab: raw as Tab, openPalette: false };
  }
  if (raw in TAB_ALIASES) {
    return { tab: TAB_ALIASES[raw], openPalette: raw === "chat" };
  }
  return { tab: "dashboard", openPalette: false };
}

export default function AdminDashboard() {
  const { user: authUser, role: userRole } = useCurrentUser();

  const handleLogout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/admin/login";
  }, []);

  const [active, setActive] = useState<Tab>(() => {
    if (typeof window !== "undefined") {
      const raw = new URLSearchParams(window.location.search).get("tab");
      return resolveTab(raw).tab;
    }
    return "dashboard";
  });
  const [paletteOpen, setPaletteOpen] = useState(() => {
    if (typeof window !== "undefined") {
      const raw = new URLSearchParams(window.location.search).get("tab");
      return resolveTab(raw).openPalette;
    }
    return false;
  });
  const [actionsBadge, setActionsBadge] = useState(0);
  const [jobIssues, setJobIssues] = useState(0);
  const [pendingApprovals, setPendingApprovals] = useState(0);
  const [researchBadge, setResearchBadge] = useState(0);
  const [cloudHealthy, setCloudHealthy] = useState(true);
  const [xHealthy, setXHealthy] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [account, setAccountRaw] = useState<XAccount>(() => {
    if (typeof window === "undefined") return "jasonsosa";
    const raw = new URLSearchParams(window.location.search).get("account");
    if (raw === "jasonsosa" || raw === "omega_memory") return raw;
    return "jasonsosa";
  });
  const setAccount = useCallback((next: XAccount) => {
    setAccountRaw(next);
    const url = new URL(window.location.href);
    url.searchParams.set("account", next);
    window.history.replaceState({}, "", url.toString());
  }, []);
  // Ambient status: powers sidebar badges, tab title, and toast notifications
  const { status: ambientStatus, events: ambientEvents, dismissEvent } = useAmbientStatus();

  // Dynamic browser tab title
  useEffect(() => {
    const n = ambientStatus.activeAgents;
    document.title = n > 0 ? `(${n}) OMEGA Admin` : "OMEGA Admin";
    return () => { document.title = "OMEGA Admin"; };
  }, [ambientStatus.activeAgents]);

  // Poll for job issues (error/late) and pending approvals to show sidebar badges
  const checkBadges = useCallback(async () => {
    try {
      const [schedRes, healthRes, appRes] = await Promise.all([
        fetch("/api/schedules"),
        fetch("/api/admin/schedule-runs/health").catch(() => null),
        fetch("/api/admin/approvals").catch(() => null),
      ]);
      if (schedRes.ok) {
        const { schedules: list } = await schedRes.json();
        const healthMap: Record<string, string[]> = healthRes?.ok
          ? (await healthRes.json()).health ?? {}
          : {};
        const { computeStatus } = await import("./jobs/jobUtils");
        let issues = 0;
        let cloudOk = true;
        let xOk = true;
        for (const s of (list ?? []) as any[]) {
          const label = s.label as string;
          const st = computeStatus(s, healthMap[label]);
          if (st === "error" || st === "late") issues++;
          const nameL = (s.name ?? "").toLowerCase();
          if ((nameL.includes("cloud") || nameL.includes("sync")) && (st === "error" || st === "late")) cloudOk = false;
          if ((nameL.includes("tweet") || nameL.includes("reply") || nameL.includes("publish") || nameL.includes("x-")) && (st === "error" || st === "late")) xOk = false;
        }
        setJobIssues(issues);
        setCloudHealthy(cloudOk);
        setXHealthy(xOk);
      }
      if (appRes?.ok) {
        const { approvals } = await appRes.json();
        setPendingApprovals(Array.isArray(approvals) ? approvals.length : 0);
      }
    } catch (err) {
      console.warn("[checkBadges] badge poll failed:", err);
    }
  }, []);

  // Run on mount + smart poll (pauses when tab hidden, refetches on return)
  useEffect(() => { checkBadges(); }, [checkBadges]);
  useSmartPoll(checkBadges, 60_000);

  // Health dots for topbar — matches preview layout: Memory, Jobs, Cloud, X
  type DotStatus = "healthy" | "warning" | "error";
  const healthDots = useMemo(() => {
    const memCount = ambientStatus.recentMemories;
    const memStatus: DotStatus = memCount > 0 ? "healthy" : "warning";
    const jobStatus: DotStatus = jobIssues === 0 ? "healthy" : "error";
    const cloudStatus: DotStatus = cloudHealthy ? "healthy" : "warning";
    const xStatus: DotStatus = xHealthy ? "healthy" : "warning";
    return [
      { label: "Memory", status: memStatus, detail: memCount > 0 ? `${memCount} recent memories` : "No recent activity" },
      { label: "Jobs", status: jobStatus, detail: jobIssues === 0 ? "All jobs running" : `${jobIssues} job${jobIssues > 1 ? "s" : ""} failing` },
      { label: "Cloud", status: cloudStatus, detail: cloudHealthy ? "Cloud sync healthy" : "Cloud sync issues" },
      { label: "X", status: xStatus, detail: xHealthy ? "X publishing healthy" : "X publishing issues" },
    ];
  }, [ambientStatus.recentMemories, jobIssues, cloudHealthy, xHealthy]);

  // Cmd+K shortcut
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const { generating, genProgress, refreshKey, handleGenerate } = useGenerateAction(account);

  const handleActionsBadge = useCallback((count: number) => setActionsBadge(count), []);

  const handleNavigate = useCallback((tab: Tab) => {
    setActive(tab);
    setMobileMenuOpen(false);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", tab);
    window.history.replaceState({}, "", url.toString());
  }, []);

  return (
    <div className="admin-scope flex h-screen overflow-hidden bg-canvas">
      <a href="#admin-main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-lg focus:bg-surface focus:border focus:border-edge focus:text-gold focus:text-sm focus:font-medium focus:shadow-lg">
        Skip to main content
      </a>
      {/* Desktop sidebar */}
      <Sidebar
        active={active}
        onNavigate={handleNavigate}
        actionsBadge={actionsBadge}
        activeAgents={ambientStatus.activeAgents}
        recentMemories={ambientStatus.recentMemories}
        hasConflicts={ambientStatus.fileConflicts.length > 0}
        sparkline={ambientStatus.sparkline}
        jobIssues={jobIssues}
        pendingApprovals={pendingApprovals}
        researchBadge={researchBadge}
        role={userRole}
      />

      {/* Mobile sidebar overlay */}
      {mobileMenuOpen && (
        <>
          <div
            className="lg:hidden fixed inset-0 z-40 bg-canvas/60 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          />
          <div role="dialog" aria-modal="true" aria-label="Navigation menu" className="lg:hidden fixed inset-y-0 left-0 z-50 w-60 bg-surface border-r border-edge flex flex-col card-enter">
            {/* Brand */}
            <div className="flex items-center justify-between h-[52px] px-4 shrink-0">
              <div className="flex items-center gap-2.5">
                <OmegaMark size={18} className="text-gold" />
                <span className="font-mono text-[13px] tracking-[0.15em] uppercase text-ink/70">
                  OMEGAMAX
                </span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                aria-label="Close menu"
                className="w-8 h-8 flex items-center justify-center rounded-md text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* All nav items for mobile */}
            <nav className="flex-1 flex flex-col px-2 pt-1 gap-0.5 overflow-y-auto">
              {getVisibleTabs(userRole).map((id) => {
                const isActive = active === id;
                return (
                  <button
                    key={id}
                    onClick={() => handleNavigate(id)}
                    className={`relative flex items-center gap-3 px-3 py-2.5 rounded-md text-[13px] font-medium transition-all ${
                      isActive
                        ? "bg-gold/[0.08] text-gold"
                        : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
                    }`}
                  >
                    {isActive && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[2.5px] h-5 rounded-r-sm bg-gold shadow-[0_0_8px_rgba(212,168,67,0.4)]" />
                    )}
                    {TAB_LABELS[id]}
                    {id === "actions" && actionsBadge > 0 && (
                      <span className="ml-auto min-w-[18px] h-[18px] bg-type-error text-white text-[10px] font-bold rounded-full flex items-center justify-center px-1">
                        {actionsBadge > 9 ? "9+" : actionsBadge}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        </>
      )}

      {/* Main content area */}
      <div className="flex-1 min-w-0 overflow-hidden lg:p-2">
        <div className="flex flex-col h-full lg:border lg:border-edge lg:rounded-xl lg:overflow-hidden bg-canvas">
        <TopBar
          onSearchOpen={() => setPaletteOpen(true)}
          onMobileMenuOpen={() => setMobileMenuOpen(true)}
          onLogout={handleLogout}
          userEmail={authUser?.email}
          userRole={userRole}
          healthDots={healthDots}
        />

        {/* Passkey setup banner */}
        <PasskeyBanner />

        {/* Content */}
        {active === "coordination" ? (
          /* Coordination gets full bleed - ReactFlow manages its own viewport */
          <div className="flex-1 min-h-0 relative z-[1]">
            <div className="absolute inset-0 flex flex-col">
              <TabErrorBoundary fallbackLabel="Coordination tab encountered an error.">
                <Suspense fallback={<TabSkeleton />}>
                  <Coordination />
                </Suspense>
              </TabErrorBoundary>
            </div>
          </div>
        ) : active === "projects" ? (
          /* Projects gets full bleed - has its own internal sidebar */
          <div className="flex-1 min-h-0 relative z-[1]">
            <div className="absolute inset-0 flex flex-col">
              <TabErrorBoundary fallbackLabel="Projects tab encountered an error.">
                <Suspense fallback={<TabSkeleton />}>
                  <Projects />
                </Suspense>
              </TabErrorBoundary>
            </div>
          </div>
        ) : (
          <main id="admin-main-content" className="flex-1 overflow-y-auto pb-16 lg:pb-4 relative z-[1]">
            <div className="admin-wash" />
            <div className={`${TAB_WIDTH_MAP[active] ?? "max-w-5xl"} mx-auto relative`}>
              <div key={active} className="admin-tab-enter">
                <TabErrorBoundary>
                <Suspense fallback={<TabSkeleton />}>
                {active === "dashboard" && (
                  <Dashboard
                    onNavigate={handleNavigate}
                    ambientStatus={ambientStatus}
                    account={account}
                    userId={authUser?.id}
                  />
                )}
                {active === "feed" && (
                  <Feed onNavigateToApprovals={() => setActive("actions")} />
                )}
                {active === "actions" && (
                  <ActionsView
                    onCountChange={handleActionsBadge}
                    onGenerate={handleGenerate}
                    generating={generating}
                    genProgress={genProgress}
                    refreshKey={refreshKey}
                    account={account}
                    onAccountChange={setAccount}
                  />
                )}
                {active === "insights" && <Insights account={account} />}
                {active === "research" && <Research />}
                {active === "docs" && <KnowledgeBase />}
                {active === "jobs" && <Schedules />}
                {active === "settings" && <Settings />}
                {active === "entities" && <Entities />}
                {active === "growth" && <GrowthTab />}
                {active === "diagnostic" && <Diagnostic />}
                </Suspense>
                </TabErrorBoundary>
              </div>
            </div>
          </main>
        )}
        </div>
      </div>

      {/* Mobile bottom nav */}
      <MobileNav active={active} onNavigate={handleNavigate} actionsBadge={actionsBadge} />

      {/* Command palette */}
      <Suspense fallback={null}>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      </Suspense>

      {/* Ambient toast notifications */}
      <Suspense fallback={null}>
        <AmbientToast events={ambientEvents} onDismiss={dismissEvent} />
      </Suspense>
    </div>
  );
}

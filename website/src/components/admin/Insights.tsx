import React, { useCallback, useEffect, useState, Suspense, lazy } from "react";
import type { InsightsData } from "./lib/types";
import WorkflowChanges from "./WorkflowChanges";
import EngagementSummary from "./EngagementSummary";
import { AIInsights } from "./insights/InsightCard";
import SummaryCards from "./insights/SummaryCards";
import Publishing from "./insights/Publishing";
import ScheduleHealth from "./insights/ScheduleHealth";
import ActivityHeatmap from "./insights/ActivityHeatmap";
import CampaignEditor from "./insights/CampaignEditor";
import WeeklyReports from "./insights/WeeklyReports";

// Lazy-load recharts-heavy components to reduce initial bundle
const MemorySystem = lazy(() => import("./insights/MemorySystem"));
const ProjectRadar = lazy(() => import("./insights/ProjectRadar"));
const RhoMetric = lazy(() => import("./insights/RhoMetric"));

interface InsightsProps {
  account?: "all" | "jasonsosa" | "omega_memory";
}

export default function Insights({ account = "all" }: InsightsProps) {
  const [data, setData] = useState<InsightsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [period, setPeriod] = useState<7 | 30 | 90>(30);
  const [menuOpen, setMenuOpen] = useState(false);

  const fetchInsights = useCallback(async (days: number, acct: string) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ period: String(days) });
      if (acct !== "all") params.set("account", acct);
      const res = await fetch(`/api/admin/insights?${params}`);
      if (!res.ok) throw new Error(`${res.status}`);
      setData(await res.json());
    } catch {
      setError("Failed to load insights");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchInsights(period, account);
  }, [period, account, fetchInsights]);

  if (loading) {
    return (
      <div className="px-5 pt-6 space-y-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="p-4 admin-card">
            <div className="skeleton h-4 w-24 rounded mb-3" />
            <div className="skeleton h-16 w-full rounded-lg" />
          </div>
        ))}
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="px-5 pt-6">
        <div className="p-4 bg-type-error/10 border border-type-error/20 rounded-lg text-[16px] text-type-error">
          {error || "No data available"}
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 pt-4 pb-8 space-y-5">
      {/* Header + period toggle */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="admin-heading">Insights</h3>
          <p className="text-[16px] text-ink-tertiary font-mono mt-0.5">
            {data.memory.total} memories
          </p>
        </div>
        <div className="relative">
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-elevated border border-edge text-ink-secondary hover:text-ink text-[14px] transition-colors touch-manipulation min-h-[44px]"
          >
            <span className="font-mono tabular-nums">Last {period}d</span>
            <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${menuOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>
          {menuOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
              <div className="absolute right-0 top-full mt-1 z-50 bg-surface border border-edge-strong rounded-lg shadow-lg py-1 min-w-[140px]">
                {([7, 30, 90] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => { setPeriod(d); setMenuOpen(false); }}
                    className={`flex items-center justify-between w-full px-3 py-2 text-[14px] transition-colors touch-manipulation ${
                      period === d ? "text-gold" : "text-ink-secondary hover:text-ink hover:bg-surface-hover"
                    }`}
                  >
                    <span>Last {d} days</span>
                    {period === d && (
                      <svg className="w-4 h-4 text-gold" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                      </svg>
                    )}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* AI Insights: highest signal, always visible at top */}
      {(data.memoryInsights || []).length > 0 && (
        <>
          <div className="admin-divider" />
          <AIInsights insights={data.memoryInsights || []} />
        </>
      )}

      {/* Summary Cards */}
      <SummaryCards summary={data.summary} memory={data.memory} content={data.content} period={period} />

      {/* Verification Rate (ρ) */}
      <div className="admin-divider" />
      <Suspense fallback={null}>
        <RhoMetric verification={data.verification} />
      </Suspense>

      {/* Memory System */}
      <div className="admin-divider" />
      <Suspense fallback={null}>
        <MemorySystem memory={data.memory} period={period} />
      </Suspense>

      {/* Workflow Changes */}
      <div className="admin-divider" />
      <WorkflowChanges />

      {/* Project Radar */}
      <div className="admin-divider" />
      <Suspense fallback={null}>
        <ProjectRadar allocation={data.allocation} projects={data.projects} unscopedCount={data.unscopedCount} />
      </Suspense>

      {/* Engagement Feedback */}
      <div className="admin-divider" />
      <EngagementSummary account={account} />

      {/* Publishing (collapsed by default) — owner-only */}
      {data.content && (
        <>
          <div className="admin-divider" />
          <Publishing content={data.content} />
        </>
      )}

      {/* Schedule Health (collapsed by default, opens if problems) — owner-only */}
      {data.schedules && (
        <>
          <div className="admin-divider" />
          <ScheduleHealth schedules={data.schedules} />
        </>
      )}

      {/* Activity Heatmap (collapsed by default) */}
      {data.heatmap && Object.keys(data.heatmap).length > 0 && (
        <>
          <div className="admin-divider" />
          <ActivityHeatmap heatmap={data.heatmap} />
        </>
      )}

      {/* Campaign Editor */}
      <div className="admin-divider" />
      <CampaignEditor account={account} />

      {/* Weekly Reports */}
      <div className="admin-divider" />
      <WeeklyReports account={account} />
    </div>
  );
}

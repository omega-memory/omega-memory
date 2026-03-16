import { useState } from "react";
import type { InsightsData, DashboardData, Tab } from "../lib/types";
import { SIGNAL_TYPES } from "../lib/constants";

// ─── Types ───────────────────────────────────────────────────

type Grade = "A" | "B" | "C" | "D" | "F";

interface GradeMetric {
  label: string;
  value: string;
  status?: "healthy" | "warning" | "error";
}

interface SystemGrade {
  system: string;
  grade: Grade;
  score: number; // 0-100
  metrics: GradeMetric[];
  suggestion?: string;
  tab?: Tab; // navigable admin tab
}

// ─── Grade helpers ───────────────────────────────────────────

const GRADE_FROM_SCORE = (s: number): Grade =>
  s >= 90 ? "A" : s >= 75 ? "B" : s >= 55 ? "C" : s >= 30 ? "D" : "F";

const GRADE_POINTS: Record<Grade, number> = { A: 4, B: 3, C: 2, D: 1, F: 0 };

const GRADE_COLORS: Record<Grade, { text: string; bg: string }> = {
  A: { text: "text-type-lesson", bg: "bg-type-lesson/15" },
  B: { text: "text-[#60a5fa]", bg: "bg-[#60a5fa]/15" },
  C: { text: "text-type-reminder", bg: "bg-type-reminder/15" },
  D: { text: "text-[#f59e0b]", bg: "bg-[#f59e0b]/15" },
  F: { text: "text-type-error", bg: "bg-type-error/15" },
};

// ─── Grading functions ───────────────────────────────────────

function gradeMemory(insights: InsightsData): SystemGrade {
  const total = insights.memory.totalAll;
  const growth = insights.summary.growthPct;
  const unscoped = insights.unscopedCount;
  const unscopedPct = total > 0 ? (unscoped / total) * 100 : 0;

  // Signal ratio: high-value types / period total
  const byType = insights.memory.byType ?? {};
  const signalCount = Object.entries(byType)
    .filter(([t]) => SIGNAL_TYPES.has(t))
    .reduce((sum, [, n]) => sum + n, 0);
  const periodTotal = insights.memory.total;
  const signalPct = periodTotal > 0 ? (signalCount / periodTotal) * 100 : 0;

  // Normalize growth: -1 is a sentinel for "no prior data", treat as neutral
  const effectiveGrowth = growth === -1 ? 0 : growth;

  let score = 0;
  if (total === 0) score = 0;
  else if (total >= 500 && effectiveGrowth >= 0 && signalPct > 70 && unscopedPct < 15) score = 95;
  else if (total >= 200 && effectiveGrowth >= -5) score = 78;
  else if (total >= 50) score = 55;
  else score = 25;

  // Fine-tune: penalize high unscoped, reward high signal
  if (unscopedPct > 20) score = Math.max(score - 10, 0);
  if (signalPct > 80) score = Math.min(score + 5, 100);
  if (effectiveGrowth < -5) score = Math.max(score - 8, 0);

  const metrics: GradeMetric[] = [
    { label: "Stored", value: total.toLocaleString(), status: total > 200 ? "healthy" : total > 50 ? "warning" : "error" },
    { label: "Growth", value: growth === -1 ? "No baseline" : `${growth >= 0 ? "+" : ""}${growth.toFixed(1)}%`, status: growth === -1 ? "healthy" : effectiveGrowth > 0 ? "healthy" : effectiveGrowth === 0 ? "warning" : "error" },
    { label: "Signal", value: `${signalPct.toFixed(0)}%`, status: signalPct > 70 ? "healthy" : signalPct > 40 ? "warning" : "error" },
    { label: "Unscoped", value: `${unscoped}`, status: unscopedPct < 15 ? "healthy" : unscopedPct < 25 ? "warning" : "error" },
  ];

  let suggestion: string | undefined;
  if (total === 0) suggestion = "No memories stored yet";
  else if (unscopedPct > 20) suggestion = `Reduce unscoped memories (${unscoped} / ${unscopedPct.toFixed(0)}%)`;
  else if (effectiveGrowth < 0) suggestion = "Negative growth trend; check if capture is active";
  else if (signalPct < 50) suggestion = "Low signal ratio; more decisions and lessons needed";

  return { system: "Memory", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion, tab: "insights" as Tab };
}

function gradeJobs(insights: InsightsData): SystemGrade {
  const enabled = (insights.schedules ?? []).filter((s) => s.enabled);
  const failing = enabled.filter((s) => s.lastStatus === "error" && s.lastRunAt);
  const overdue = enabled.filter((s) => s.overdue);
  const unhealthy = new Set([...failing.map((f) => f.id), ...overdue.map((o) => o.id)]);
  const healthy = enabled.length - unhealthy.size;

  let score = 0;
  if (enabled.length === 0) score = 0;
  else if (failing.length === 0 && overdue.length === 0) score = 95;
  else if (failing.length === 0 && overdue.length > 0) score = 78;
  else if (failing.length === 1) score = 50;
  else if (failing.length < enabled.length) score = 25;
  else score = 0;

  const lastRun = enabled
    .filter((s) => s.lastRunAt)
    .sort((a, b) => new Date(b.lastRunAt!).getTime() - new Date(a.lastRunAt!).getTime())[0];
  const lastRunAgo = lastRun?.lastRunAt
    ? formatAgo(new Date(lastRun.lastRunAt))
    : "never";

  const metrics: GradeMetric[] = [
    { label: "Healthy", value: `${healthy}/${enabled.length}`, status: failing.length === 0 ? "healthy" : "error" },
    { label: "Overdue", value: `${overdue.length}`, status: overdue.length === 0 ? "healthy" : "warning" },
    { label: "Last sync", value: lastRunAgo },
  ];

  let suggestion: string | undefined;
  if (enabled.length === 0) suggestion = "No jobs enabled";
  else if (failing.length > 0) suggestion = `Fix failing: ${failing.map((f) => f.name).join(", ")}`;
  else if (overdue.length > 0) suggestion = `${overdue.length} job${overdue.length > 1 ? "s" : ""} overdue; check scheduler`;

  return { system: "Jobs", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion, tab: "jobs" as Tab };
}

function gradeCloudSync(insights: InsightsData): SystemGrade {
  const totalAll = insights.memory.totalAll;
  const cloud = insights.memory.totalCloud;
  const cloudAvailable = cloud != null;
  const cloudCount = cloud ?? 0;
  // Compare all-time totals: totalAll is all-time Supabase count,
  // cloud is the same source. Gap reflects query health, not local↔cloud.
  const gapPct = totalAll > 0 ? (Math.abs(totalAll - cloudCount) / totalAll) * 100 : 0;

  let score = 0;
  if (!cloudAvailable) score = 0;
  else if (cloudCount === 0) score = 0;
  else if (gapPct < 5) score = 95;
  else if (gapPct < 15) score = 78;
  else if (gapPct < 30) score = 55;
  else score = 25;

  const metrics: GradeMetric[] = [
    { label: "Cloud", value: cloudCount.toLocaleString(), status: cloudCount > 0 ? "healthy" : "error" },
    { label: "Total", value: totalAll.toLocaleString() },
    { label: "Gap", value: `${gapPct.toFixed(1)}%`, status: gapPct < 5 ? "healthy" : gapPct < 15 ? "warning" : "error" },
    { label: "Status", value: !cloudAvailable ? "Query failed" : cloudCount > 0 ? "Connected" : "Disconnected", status: !cloudAvailable ? "error" : cloudCount > 0 ? "healthy" : "error" },
  ];

  let suggestion: string | undefined;
  if (!cloudAvailable) suggestion = "Cloud count query failed; check Supabase connection";
  else if (cloudCount === 0) suggestion = "No cloud connection detected";
  else if (gapPct > 15) suggestion = `Sync gap growing (${gapPct.toFixed(0)}%); check cloud sync`;
  else if (gapPct > 5) suggestion = "Small sync gap; monitor for drift";

  return { system: "Cloud Sync", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion };
}

function gradePublishing(insights: InsightsData): SystemGrade {
  if (!insights.content) return { system: "Publishing", grade: "A" as Grade, score: 100, metrics: [], tab: "insights" as Tab };
  const tweets = insights.content.tweets;
  const linkedin = insights.content.linkedin;
  const published = tweets.published + linkedin.published;
  const failed = tweets.failed + linkedin.failed;
  const total = published + failed;
  const rate = total > 0 ? (published / total) * 100 : 0;
  const queue = tweets.queueDepth + linkedin.queueDepth;

  let score = 0;
  if (total === 0) score = 0;
  else if (rate > 95 && failed === 0) score = 95;
  else if (rate > 80) score = 78;
  else if (rate > 60) score = 55;
  else if (rate > 40) score = 30;
  else score = 10;

  const metrics: GradeMetric[] = [
    { label: "Published", value: `${published}`, status: published > 0 ? "healthy" : "error" },
    { label: "Failed", value: `${failed}`, status: failed === 0 ? "healthy" : "error" },
    { label: "Success", value: `${rate.toFixed(0)}%`, status: rate > 80 ? "healthy" : rate > 60 ? "warning" : "error" },
    { label: "Queue", value: `${queue}`, status: queue < 10 ? "healthy" : "warning" },
  ];

  let suggestion: string | undefined;
  if (total === 0) suggestion = "No published content yet";
  else if (failed > 0) suggestion = `Investigate ${failed} failed post${failed > 1 ? "s" : ""}; retry or remove`;
  else if (queue > 15) suggestion = `Queue depth is ${queue}; consider clearing stale items`;

  return { system: "Publishing", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion };
}

function gradeEngagement(dashboard: DashboardData): SystemGrade {
  const perf = dashboard.performance;
  if (!perf) {
    return {
      system: "Engagement",
      grade: "F",
      score: 0,
      metrics: [{ label: "Data", value: "None", status: "error" }],
      suggestion: "No performance data available",
    };
  }

  const rate = perf.overall.avgEngagementRate;
  const trend = perf.overall.recentTrend;
  const tracked = perf.overall.totalTracked;

  let score = 0;
  if (rate > 3 && trend === "improving") score = 95;
  else if (rate > 2) score = 78;
  else if (rate > 1 && trend === "improving") score = 78;
  else if (rate > 1 && trend === "steady") score = 55;
  else if (trend === "declining") score = 25;
  else if (rate < 1) score = 25;
  else score = 40;

  const trendLabel = trend === "improving" ? "Improving" : trend === "declining" ? "Declining" : "Steady";

  const metrics: GradeMetric[] = [
    { label: "Avg rate", value: `${rate.toFixed(1)}%`, status: rate > 2 ? "healthy" : rate > 1 ? "warning" : "error" },
    { label: "Trend", value: trendLabel, status: trend === "improving" ? "healthy" : trend === "steady" ? "warning" : "error" },
    { label: "Tracked", value: `${tracked}` },
  ];

  let suggestion: string | undefined;
  if (trend === "declining") suggestion = "Engagement declining; try thread format or different topics";
  else if (rate < 1.5) suggestion = "Low engagement; experiment with posting times and formats";
  else if (trend === "improving") suggestion = "Current strategy is working; maintain cadence";

  return { system: "Engagement", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion };
}

function gradeDistribution(dashboard: DashboardData): SystemGrade {
  const gh = dashboard.github;
  const pypi = dashboard.pypi;
  const tw = dashboard.twitter;

  const connected = [gh, pypi, tw].filter(Boolean).length;
  const stars = gh?.stars ?? 0;
  const weeklyDl = pypi?.weeklyDownloads ?? 0;
  const followers = tw?.followers ?? 0;

  let score = 0;
  if (connected === 0) score = 0;
  else if (connected === 3 && stars > 50 && weeklyDl > 500) score = 95;
  else if (connected === 3) score = 78;
  else if (connected === 2) score = 55;
  else score = 25;

  const fmtK = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
  const xAccts = tw?.accounts;

  const metrics: GradeMetric[] = [
    { label: "GitHub", value: gh ? `${stars} stars` : "N/A", status: gh ? "healthy" : "error" },
    { label: "PyPI", value: pypi ? `${fmtK(weeklyDl)}/wk` : "N/A", status: pypi ? "healthy" : "error" },
    ...(xAccts?.jasonsosa && xAccts?.omega_memory
      ? [
          { label: "@jasonsosa", value: `${fmtK(xAccts.jasonsosa.followers)}`, status: "healthy" as const },
          { label: "@omega_memory", value: `${fmtK(xAccts.omega_memory.followers)}`, status: "healthy" as const },
        ]
      : [{ label: "X", value: tw ? `${fmtK(followers)} followers` : "N/A", status: (tw ? "healthy" : "error") as "healthy" | "error" }]),
  ];

  let suggestion: string | undefined;
  if (connected < 3) {
    const missing = [!gh && "GitHub", !pypi && "PyPI", !tw && "X"].filter(Boolean);
    suggestion = `Connect missing: ${missing.join(", ")}`;
  } else if (weeklyDl < 100) suggestion = "Downloads low; consider a blog post or announcement";
  else if (stars < 20) suggestion = "Stars low; engage with community and share in forums";

  return { system: "Distribution", grade: GRADE_FROM_SCORE(score), score, metrics, suggestion };
}

// ─── Time formatting ─────────────────────────────────────────

function formatAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

// ─── Sub-components ──────────────────────────────────────────

function GradeBadge({ grade }: { grade: Grade }) {
  const colors = GRADE_COLORS[grade];
  return (
    <span
      className={`inline-flex items-center justify-center w-8 h-8 rounded-full admin-text-body font-bold ${colors.text} ${colors.bg}`}
    >
      {grade}
    </span>
  );
}

const STATUS_DOT: Record<string, string> = {
  healthy: "bg-type-lesson",
  warning: "bg-type-reminder",
  error: "bg-type-error",
};

function GradeRow({
  grade,
  isOpen,
  onToggle,
  onNavigate,
}: {
  grade: SystemGrade;
  isOpen: boolean;
  onToggle: () => void;
  onNavigate?: (tab: Tab) => void;
}) {
  const inlineMetrics = grade.metrics
    .slice(0, 3)
    .map((m) => `${m.value} ${m.label.toLowerCase()}`)
    .join(" · ");

  return (
    <div className="py-2.5">
      <button onClick={onToggle} className="flex items-center gap-3 w-full text-left">
        <GradeBadge grade={grade.grade} />
        <span className="font-medium text-ink admin-text-body w-20 sm:w-28 shrink-0">{grade.system}</span>
        <span className="text-ink-secondary admin-text-caption font-mono tabular-nums truncate flex-1">
          {inlineMetrics}
        </span>
        <div className="flex items-center gap-1.5 shrink-0">
          {grade.tab && onNavigate && (
            <button
              onClick={(e) => { e.stopPropagation(); onNavigate(grade.tab!); }}
              className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-gold hover:bg-gold/10 transition-colors"
              title={`Go to ${grade.system}`}
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
            </button>
          )}
          <svg
            className={`w-4 h-4 text-ink-tertiary transition-transform duration-150 ${isOpen ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
          </svg>
        </div>
      </button>
      {grade.suggestion && !isOpen && (
        <p className="ml-11 mt-1 admin-text-body text-ink-tertiary italic">{grade.suggestion}</p>
      )}
      <div
        className="grid transition-[grid-template-rows] duration-150 ease-out"
        style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="ml-11 mt-2 py-2.5 px-3 rounded-lg bg-surface-elevated/50 border border-edge-subtle space-y-2">
            {grade.metrics.map((m) => (
              <div key={m.label} className="flex items-center justify-between gap-3 admin-text-caption">
                <span className="flex items-center gap-1.5 text-ink-secondary">
                  {m.status && (
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${STATUS_DOT[m.status]}`} />
                  )}
                  {m.label}
                </span>
                <span className="text-ink font-medium tabular-nums font-mono">{m.value}</span>
              </div>
            ))}
            {grade.suggestion && (
              <div className="pt-1.5 mt-1.5 border-t border-edge-subtle">
                <p className="admin-text-caption text-gold italic">{grade.suggestion}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────

interface Props {
  insights: InsightsData;
  dashboard: DashboardData;
  onNavigate?: (tab: Tab) => void;
}

export default function ReportCard({ insights, dashboard, onNavigate }: Props) {
  const [expandedRow, setExpandedRow] = useState<number>(-1);

  const grades: SystemGrade[] = [
    gradeMemory(insights),
    gradeJobs(insights),
    gradeCloudSync(insights),
    gradePublishing(insights),
    gradeEngagement(dashboard),
    gradeDistribution(dashboard),
  ];

  // Sort worst-first: surfaces what needs attention at the top
  grades.sort((a, b) => a.score - b.score);

  const gpa = grades.reduce((sum, g) => sum + GRADE_POINTS[g.grade], 0) / grades.length;

  return (
    <div className="admin-card-quiet p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="admin-section-label">Report Card</span>
        <span className="admin-text-body font-mono tabular-nums text-ink-secondary">
          GPA: <span className="text-ink font-semibold">{gpa.toFixed(1)}</span>
        </span>
      </div>
      <div className="divide-y divide-edge-subtle">
        {grades.map((g, i) => (
          <GradeRow
            key={g.system}
            grade={g}
            isOpen={expandedRow === i}
            onToggle={() => setExpandedRow(expandedRow === i ? -1 : i)}
            onNavigate={onNavigate}
          />
        ))}
      </div>
    </div>
  );
}

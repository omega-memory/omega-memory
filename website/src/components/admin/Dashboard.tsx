import { useCallback, useEffect, useMemo, useState } from "react";
import type { InsightsData, DashboardData, Tab, ProjectRadarCard } from "./lib/types";
import type { AmbientStatus } from "./hooks/useAmbientStatus";
import { useProjects } from "./hooks/useProjects";
import { buildSuggestions } from "./dashboardUtils";
import { SIGNAL_TYPES } from "./lib/constants";
import AlertCard, { mapProblemToAlert } from "./dashboard/AlertCard";
import DashboardRawData from "./dashboard/DashboardRawData";

// ─── Module-level cache ──────────────────────────────────────
let _cache: {
  insights: InsightsData;
  dashboard: DashboardData;
  ts: number;
  userId: string;
} | null = null;

const CACHE_TTL = 60_000;

// ─── Helpers ─────────────────────────────────────────────────

function timeAgo(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function fmtK(n: number) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

type Grade = "A" | "B" | "C" | "D" | "F";
const GRADE_FROM_SCORE = (s: number): Grade =>
  s >= 90 ? "A" : s >= 75 ? "B" : s >= 55 ? "C" : s >= 30 ? "D" : "F";
const GRADE_COLORS: Record<Grade, string> = {
  A: "#34d399", B: "#60a5fa", C: "#fbbf24", D: "#f59e0b", F: "#f06060",
};

type Trend = "up" | "down" | "flat";

interface KpiTile {
  label: string;
  value: string;
  trend: Trend;
  delta: string;
  deltaContext?: string;
  tooltip: string;
  sparkline?: number[];
}

interface MetricChip {
  label: string;
  value: string;
  trend?: Trend;
  tooltip?: string;
}

interface ProjectCard {
  name: string;
  grade: Grade;
  gradeColor: string;
  score: number;
  momentum: "up" | "steady" | "cooling" | "stalled";
  momentumDelta?: string;
  sessionCount: number;
  completed: { text: string; signal?: boolean; time: string }[];
  active: { text: string; agent?: string }[];
  attention: { text: string; urgent?: boolean }[];
  metrics: MetricChip[];
  recentDecision?: string;
}

interface Problem {
  text: string;
  urgent: boolean;
}

// ─── Data mapping ────────────────────────────────────────────

function buildKpis(insights: InsightsData, dashboard: DashboardData): KpiTile[] {
  const totalMem = insights.memory.totalAll;
  const growth = insights.summary.growthPct;
  const growthLabel = growth === -1 ? "" : `${growth >= 0 ? "+" : ""}${growth.toFixed(0)}%`;
  const sessions = insights.summary.activeSessions;
  const published = insights.summary.contentPublished;
  const engRate = dashboard.performance?.overall?.avgEngagementRate ?? 0;
  const engTrend = dashboard.performance?.overall?.recentTrend;
  const stars = dashboard.github?.stars ?? 0;
  const downloads = dashboard.pypi?.weeklyDownloads ?? 0;

  // Build sparkline from byDay data (last 14 days)
  const byDay = insights.memory.byDay ?? {};
  const sortedDays = Object.keys(byDay).sort();
  const memSparkline = sortedDays.slice(-14).map(d => byDay[d]);

  return [
    { label: "Memories", value: totalMem.toLocaleString(), trend: (growth > 0 ? "up" : growth < 0 ? "down" : "flat") as Trend, delta: growthLabel, deltaContext: growth !== -1 ? `vs prior ${insights.period}d` : undefined, tooltip: `Total stored memories. ${growthLabel} vs prior period.`, sparkline: memSparkline.length >= 3 ? memSparkline : undefined },
    { label: "Sessions", value: `${sessions}`, trend: (sessions > 0 ? "up" : "flat") as Trend, delta: "", deltaContext: `last ${insights.period}d`, tooltip: `Active agent sessions in the last ${insights.period} days.` },
    { label: "Published", value: `${published}`, trend: (published > 0 ? "up" : "flat") as Trend, delta: "", deltaContext: `last ${insights.period}d`, tooltip: `Content pieces published in the last ${insights.period} days.` },
    { label: "Engagement", value: `${engRate.toFixed(1)}%`, trend: (engTrend === "improving" ? "up" : engTrend === "declining" ? "down" : "flat") as Trend, delta: engTrend === "improving" ? "improving" : engTrend === "declining" ? "declining" : "", deltaContext: "recent trend", tooltip: "Average engagement rate (likes+replies+RTs / impressions)." },
    { label: "Stars + Downloads", value: `${stars} / ${fmtK(downloads)}`, trend: (stars > 0 || downloads > 0 ? "up" : "flat") as Trend, delta: "", deltaContext: "weekly", tooltip: "GitHub stars and PyPI weekly downloads." },
  ];
}

function buildProjects(insights: InsightsData, dashboard: DashboardData, pausedIds: Set<string>): ProjectCard[] {
  const cards: ProjectCard[] = [];

  // ── OMEGA Core ─────────────────────────────
  const totalMem = insights.memory.totalAll;
  const growth = insights.summary.growthPct;
  const cloudCount = insights.memory.totalCloud ?? 0;
  const byType = insights.memory.byType ?? {};
  const signalCount = Object.entries(byType).filter(([t]) => SIGNAL_TYPES.has(t)).reduce((s, [, n]) => s + n, 0);
  const periodTotal = insights.memory.total;
  const signalPct = periodTotal > 0 ? (signalCount / periodTotal) * 100 : 0;
  const effectiveGrowth = growth === -1 ? 0 : growth;

  let coreScore = 0;
  if (totalMem === 0) coreScore = 0;
  else if (totalMem >= 500 && effectiveGrowth >= 0 && signalPct > 70) coreScore = 95;
  else if (totalMem >= 200 && effectiveGrowth >= -5) coreScore = 78;
  else if (totalMem >= 50) coreScore = 55;
  else coreScore = 25;

  const coreGrade = GRADE_FROM_SCORE(coreScore);

  // Find OMEGA project from radar
  const omegaRadar = insights.projects.find(p => p.displayName.toLowerCase().includes("omega")) || insights.projects[0];

  const coreCompleted: ProjectCard["completed"] = [];
  if (periodTotal > 0) coreCompleted.push({ text: `Stored ${periodTotal.toLocaleString()} memories from ${insights.summary.activeSessions} sessions`, signal: true, time: `${insights.period}d` });
  if (cloudCount > 0) coreCompleted.push({ text: `${cloudCount.toLocaleString()} memories synced to cloud`, signal: cloudCount > totalMem * 0.9, time: "latest" });
  if (signalCount > 0) coreCompleted.push({ text: `${signalCount} high-signal memories (${signalPct.toFixed(0)}% signal ratio)`, time: `${insights.period}d` });

  const coreAttention: ProjectCard["attention"] = [];
  const unscoped = insights.unscopedCount;
  if (unscoped > 0 && (unscoped / totalMem) > 0.15) coreAttention.push({ text: `${unscoped} unscoped memories (${((unscoped / totalMem) * 100).toFixed(0)}%); consider scoping`, urgent: (unscoped / totalMem) > 0.25 });
  if (effectiveGrowth < -5) coreAttention.push({ text: "Negative growth trend; check if capture is active", urgent: true });

  cards.push({
    name: "OMEGA Core",
    grade: coreGrade, gradeColor: GRADE_COLORS[coreGrade], score: coreScore,
    momentum: omegaRadar?.momentum ?? (effectiveGrowth > 5 ? "up" : effectiveGrowth >= 0 ? "steady" : "cooling"),
    momentumDelta: growth === -1 ? undefined : `${growth >= 0 ? "+" : ""}${growth.toFixed(0)}%`,
    sessionCount: insights.summary.activeSessions,
    completed: coreCompleted,
    active: [],
    attention: coreAttention,
    metrics: [
      { label: "Memories", value: totalMem.toLocaleString(), trend: effectiveGrowth > 0 ? "up" : effectiveGrowth < 0 ? "down" : "flat", tooltip: "Total stored memories across all entities" },
      { label: "Signal", value: `${signalPct.toFixed(0)}%`, trend: signalPct > 70 ? "up" : signalPct > 40 ? "flat" : "down", tooltip: "High-value types (decisions, lessons) / total" },
      { label: "Cloud", value: cloudCount.toLocaleString(), trend: cloudCount > 0 ? "up" : "down", tooltip: "Memories synced to Supabase" },
    ],
    recentDecision: omegaRadar?.recentDecisions?.[0]?.content,
  });

  // ── Jobs ───────────────────────────────────
  const enabled = (insights.schedules ?? []).filter(s => s.enabled);
  const failing = enabled.filter(s => s.lastStatus === "error" && s.lastRunAt);
  const overdue = enabled.filter(s => s.overdue);
  const healthy = enabled.length - new Set([...failing.map(f => f.id), ...overdue.map(o => o.id)]).size;
  let jobScore = enabled.length === 0 ? 0 : failing.length === 0 && overdue.length === 0 ? 95 : failing.length === 0 && overdue.length > 0 ? 78 : failing.length === 1 ? 50 : failing.length < enabled.length ? 25 : 0;
  const jobGrade = GRADE_FROM_SCORE(jobScore);

  const jobCompleted: ProjectCard["completed"] = [];
  if (healthy > 0) jobCompleted.push({ text: `${healthy}/${enabled.length} jobs running successfully`, signal: failing.length === 0, time: "latest" });
  const jobAttention: ProjectCard["attention"] = [];
  if (failing.length > 0) jobAttention.push({ text: `${failing.length} job${failing.length > 1 ? "s" : ""} failing: ${failing.map(f => f.name).join(", ")}`, urgent: true });
  if (overdue.length > 0) jobAttention.push({ text: `${overdue.length} job${overdue.length > 1 ? "s" : ""} overdue`, urgent: overdue.length > 2 });

  cards.push({
    name: "Scheduled Jobs",
    grade: jobGrade, gradeColor: GRADE_COLORS[jobGrade], score: jobScore,
    momentum: failing.length > 0 ? "stalled" : overdue.length > 0 ? "cooling" : "steady",
    sessionCount: 0,
    completed: jobCompleted, active: [], attention: jobAttention,
    metrics: [
      { label: "Healthy", value: `${healthy}/${enabled.length}`, trend: failing.length === 0 ? "up" : "down", tooltip: "Jobs with no errors or overdue status" },
      { label: "Failing", value: `${failing.length}`, trend: failing.length === 0 ? "flat" : "down", tooltip: "Jobs with error status" },
      { label: "Overdue", value: `${overdue.length}`, trend: overdue.length === 0 ? "flat" : "down", tooltip: "Jobs past their expected run time" },
    ],
  });

  // ── Social / Publishing ────────────────────
  const tw = dashboard.twitter;
  const totalFollowers = tw?.accounts
    ? (tw.accounts.jasonsosa?.followers ?? 0) + (tw.accounts.omega_memory?.followers ?? 0)
    : tw?.followers ?? 0;
  const tweets = insights.content?.tweets;
  const linkedin = insights.content?.linkedin;
  const published = (tweets?.published ?? 0) + (linkedin?.published ?? 0);
  const failed = (tweets?.failed ?? 0) + (linkedin?.failed ?? 0);
  const engRate = dashboard.performance?.overall?.avgEngagementRate ?? 0;
  const engTrend = dashboard.performance?.overall?.recentTrend ?? "steady";

  let socialScore = published === 0 && failed === 0 ? 0 : failed > 0 && engRate < 1.5 ? 25 : engRate > 2 ? 78 : 55;
  if (engTrend === "improving" && engRate > 2) socialScore = 95;
  const socialGrade = GRADE_FROM_SCORE(socialScore);

  const socialCompleted: ProjectCard["completed"] = [];
  if (published > 0) socialCompleted.push({ text: `Published ${published} post${published > 1 ? "s" : ""}`, signal: true, time: `${insights.period}d` });
  if (tw) socialCompleted.push({ text: `${totalFollowers.toLocaleString()} total followers across accounts`, time: "current" });
  const socialAttention: ProjectCard["attention"] = [];
  if (failed > 0) socialAttention.push({ text: `${failed} post${failed > 1 ? "s" : ""} failed to publish`, urgent: true });
  if (engTrend === "declining") socialAttention.push({ text: `Engagement declining (${engRate.toFixed(1)}% avg)`, urgent: true });

  const bestType = dashboard.performance?.byContentType?.sort((a, b) => b.avgEngagementRate - a.avgEngagementRate)[0];

  cards.push({
    name: "Social",
    grade: socialGrade, gradeColor: GRADE_COLORS[socialGrade], score: socialScore,
    momentum: engTrend === "improving" ? "up" : engTrend === "declining" ? "cooling" : "steady",
    momentumDelta: engTrend === "improving" ? "improving" : engTrend === "declining" ? "declining" : undefined,
    sessionCount: 0,
    completed: socialCompleted, active: [], attention: socialAttention,
    metrics: [
      { label: "Engagement", value: `${engRate.toFixed(1)}%`, trend: engTrend === "improving" ? "up" : engTrend === "declining" ? "down" : "flat", tooltip: "Avg engagement rate across tracked posts" },
      { label: "Followers", value: totalFollowers.toLocaleString(), trend: "up", tooltip: "Combined follower count" },
      ...(bestType ? [{ label: "Best format", value: bestType.contentType, trend: "up" as Trend, tooltip: `Highest avg engagement: ${bestType.avgEngagementRate.toFixed(1)}%` }] : []),
      { label: "Published", value: `${published}`, trend: published > 0 ? "up" : "flat", tooltip: `Posts published in ${insights.period}d` },
    ],
    recentDecision: bestType ? `Best performing: ${bestType.contentType} (${bestType.avgEngagementRate.toFixed(1)}% avg)` : undefined,
  });

  // ── Cloud Sync ─────────────────────────────
  const gapPct = totalMem > 0 ? (Math.abs(totalMem - cloudCount) / totalMem) * 100 : 0;
  let cloudScore = cloudCount === 0 ? 0 : gapPct < 5 ? 95 : gapPct < 15 ? 78 : gapPct < 30 ? 55 : 25;
  const cloudGrade = GRADE_FROM_SCORE(cloudScore);

  cards.push({
    name: "Cloud Sync",
    grade: cloudGrade, gradeColor: GRADE_COLORS[cloudGrade], score: cloudScore,
    momentum: gapPct < 5 ? "steady" : gapPct < 15 ? "cooling" : "stalled",
    momentumDelta: gapPct > 0 ? `${gapPct.toFixed(1)}% gap` : undefined,
    sessionCount: 0,
    completed: cloudCount > 0 ? [{ text: `${cloudCount.toLocaleString()} of ${totalMem.toLocaleString()} memories synced`, signal: gapPct < 5, time: "latest" }] : [],
    active: [], attention: gapPct > 15 ? [{ text: `Sync gap growing (${gapPct.toFixed(0)}%); check cloud sync`, urgent: gapPct > 25 }] : [],
    metrics: [
      { label: "Sync rate", value: totalMem > 0 ? `${(100 - gapPct).toFixed(1)}%` : "N/A", trend: gapPct < 5 ? "up" : gapPct < 15 ? "flat" : "down", tooltip: "Memories synced to Supabase / total" },
      { label: "Cloud", value: cloudCount.toLocaleString(), tooltip: "Total memories in Supabase" },
      { label: "Gap", value: `${gapPct.toFixed(1)}%`, trend: gapPct < 5 ? "flat" : "down", tooltip: "Difference between local and cloud counts" },
    ],
  });

  // ── Distribution ───────────────────────────
  const gh = dashboard.github;
  const pypi = dashboard.pypi;
  const stars = gh?.stars ?? 0;
  const weeklyDl = pypi?.weeklyDownloads ?? 0;
  const connected = [gh, pypi, tw].filter(Boolean).length;

  let distScore = connected === 0 ? 0 : connected === 3 && stars > 50 ? 95 : connected === 3 ? 78 : connected === 2 ? 55 : 25;
  const distGrade = GRADE_FROM_SCORE(distScore);

  cards.push({
    name: "Distribution",
    grade: distGrade, gradeColor: GRADE_COLORS[distGrade], score: distScore,
    momentum: weeklyDl > 500 ? "up" : weeklyDl > 100 ? "steady" : "cooling",
    sessionCount: 0,
    completed: [
      ...(gh ? [{ text: `${stars} GitHub stars`, signal: true, time: "current" }] : []),
      ...(pypi ? [{ text: `${fmtK(weeklyDl)} PyPI downloads/week`, signal: weeklyDl > 500, time: "current" }] : []),
    ],
    active: [], attention: connected < 3 ? [{ text: `${3 - connected} integration${3 - connected > 1 ? "s" : ""} not connected`, urgent: false }] : [],
    metrics: [
      { label: "Stars", value: `${stars}`, trend: stars > 0 ? "up" : "flat", tooltip: "GitHub stars" },
      { label: "Downloads/wk", value: fmtK(weeklyDl), trend: weeklyDl > 100 ? "up" : "flat", tooltip: "PyPI weekly downloads" },
      { label: "Channels", value: `${connected}/3`, trend: connected === 3 ? "up" : "flat", tooltip: "Connected: GitHub, PyPI, X" },
    ],
  });

  // ── Project Radar cards ────────────────────
  for (const p of insights.projects) {
    // Skip if we already covered it above or if paused
    const nameL = p.displayName.toLowerCase();
    if (nameL.includes("omega") || pausedIds.has(nameL)) continue;
    if (!p.isActive && p.daysSinceLastSession > 14) continue;
    // Skip stalled/inactive projects — they're noise on the exec dashboard
    if (p.momentum === "stalled" || p.momentum === "cooling") continue;

    const pScore = p.momentum === "up" ? 85 : p.momentum === "steady" ? 70 : p.momentum === "cooling" ? 45 : 20;
    const pGrade = GRADE_FROM_SCORE(pScore);

    cards.push({
      name: p.displayName,
      grade: pGrade, gradeColor: GRADE_COLORS[pGrade], score: pScore,
      momentum: p.momentum,
      momentumDelta: p.momentumDelta !== 0 ? `${p.momentumDelta >= 0 ? "+" : ""}${p.momentumDelta}%` : undefined,
      sessionCount: p.sessionCount,
      completed: [
        ...(p.taskCompletedCount > 0 ? [{ text: `${p.taskCompletedCount} tasks completed`, signal: true, time: `${insights.period}d` }] : []),
        ...(p.decisionCount > 0 ? [{ text: `${p.decisionCount} decisions made`, time: `${insights.period}d` }] : []),
      ],
      active: p.isActive ? [{ text: p.narrative || "Active" }] : [],
      attention: p.alerts.map(a => ({ text: a, urgent: a.toLowerCase().includes("block") || a.toLowerCase().includes("fail") })),
      metrics: [
        { label: "Sessions", value: `${p.sessionCount}`, tooltip: `Sessions in ${insights.period}d` },
        { label: "Decisions", value: `${p.decisionCount}`, tooltip: "Key decisions recorded" },
      ],
      recentDecision: p.recentDecisions?.[0]?.content,
    });
  }

  return cards;
}

function buildProblems(insights: InsightsData, dashboard: DashboardData, pausedIds: Set<string>): Problem[] {
  const problems: Problem[] = [];
  const failing = (insights.schedules ?? []).filter(s => s.enabled && s.lastStatus === "error" && s.lastRunAt);
  if (failing.length > 0) problems.push({ text: `${failing.length} job${failing.length > 1 ? "s" : ""} failing: ${failing.map(f => f.name).join(", ")}`, urgent: true });

  const contentFailed = (insights.content?.tweets?.failed ?? 0) + (insights.content?.linkedin?.failed ?? 0);
  if (contentFailed > 0) problems.push({ text: `${contentFailed} post${contentFailed > 1 ? "s" : ""} failed to publish`, urgent: true });

  if (dashboard.performance?.overall?.recentTrend === "declining") {
    problems.push({ text: `Engagement rate declining (${dashboard.performance.overall.avgEngagementRate.toFixed(1)}% avg)`, urgent: true });
  }

  const cloudCount = insights.memory.totalCloud ?? 0;
  const totalMem = insights.memory.totalAll;
  const gapPct = totalMem > 0 ? (Math.abs(totalMem - cloudCount) / totalMem) * 100 : 0;
  if (gapPct > 15) problems.push({ text: `Cloud sync gap at ${gapPct.toFixed(0)}%; ${Math.abs(totalMem - cloudCount)} memories out of sync`, urgent: gapPct > 25 });
  if (cloudCount === 0) problems.push({ text: "Cloud sync has no data; check Supabase connection", urgent: true });

  const overdue = (insights.schedules ?? []).filter(s => s.enabled && s.overdue);
  if (overdue.length > 0) problems.push({ text: `${overdue.length} job${overdue.length > 1 ? "s" : ""} overdue`, urgent: false });

  // Suggestions from buildSuggestions that aren't already covered
  const suggestions = buildSuggestions(dashboard, insights, pausedIds);
  for (const s of suggestions) {
    if (!problems.some(p => p.text.includes(s.text.slice(0, 30)))) {
      problems.push({ text: s.text, urgent: false });
    }
  }

  return problems;
}

// ─── SVG Icons ───────────────────────────────────────────────

function TrendIcon({ trend }: { trend: Trend }) {
  if (trend === "up") return <svg className="w-3 h-3 text-type-lesson" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" /></svg>;
  if (trend === "down") return <svg className="w-3 h-3 text-type-error" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 4.5l-15 15m0 0h11.25M4.5 19.5V8.25" /></svg>;
  return <svg className="w-3 h-3 text-ink-faint" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14" /></svg>;
}

function ChevronIcon({ open }: { open: boolean }) {
  return <svg className={`w-4 h-4 text-ink-tertiary transition-transform duration-150 ${open ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" /></svg>;
}

function CheckIcon() {
  return <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>;
}

function SpinnerIcon() {
  return <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182M2.985 19.644l3.181-3.182" /></svg>;
}

function AlertIcon() {
  return <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" /></svg>;
}

// ─── Tooltip ─────────────────────────────────────────────────

function Tip({ text, children, position = "top" }: { text: string; children: React.ReactNode; position?: "top" | "bottom" }) {
  const isBottom = position === "bottom";
  return (
    <span className="relative group/tip inline-flex">
      {children}
      <span className={`pointer-events-none absolute left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-lg bg-[#1a1b20] border border-white/[0.08] admin-text-body text-ink-secondary leading-snug whitespace-nowrap opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150 z-[100] shadow-lg shadow-black/30 ${isBottom ? "top-full mt-2" : "bottom-full mb-2"}`}>
        {text}
        <span className={`absolute left-1/2 -translate-x-1/2 w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent ${isBottom ? "bottom-full border-b-[5px] border-b-[#1a1b20]" : "top-full -mt-px border-t-[5px] border-t-[#1a1b20]"}`} />
      </span>
    </span>
  );
}

// ─── GradeArc ────────────────────────────────────────────────

function GradeArc({ score, color, size = 44 }: { score: number; color: string; size?: number }) {
  const r = (size - 6) / 2;
  const circ = 2 * Math.PI * r;
  const prog = (score / 100) * circ;
  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth={3} />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={3}
        strokeDasharray={`${prog} ${circ}`} strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 4px ${color}40)` }} />
    </svg>
  );
}

// ─── Momentum config ─────────────────────────────────────────

const MOMENTUM_STYLE: Record<string, { label: string; cls: string }> = {
  up: { label: "Rising", cls: "text-type-lesson bg-type-lesson/[0.06] border-type-lesson/[0.1]" },
  steady: { label: "Steady", cls: "text-[#60a5fa] bg-[#60a5fa]/[0.06] border-[#60a5fa]/[0.1]" },
  cooling: { label: "Cooling", cls: "text-type-reminder bg-type-reminder/[0.06] border-type-reminder/[0.1]" },
  stalled: { label: "Stalled", cls: "text-type-error bg-type-error/[0.06] border-type-error/[0.1]" },
};

// ─── KPI Detail Panels ──────────────────────────────────────

function KpiDetailMemories({ insights }: { insights: InsightsData }) {
  const byType = Object.entries(insights.memory.byType ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const cloudCount = insights.memory.totalCloud ?? 0;
  const totalAll = insights.memory.totalAll;
  const periodTotal = insights.memory.total;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <span className="admin-section-label">By Type ({insights.period}d)</span>
        <div className="mt-2 space-y-1">
          {byType.map(([type, count]) => {
            const pct = periodTotal > 0 ? (count / periodTotal) * 100 : 0;
            return (
              <div key={type} className="flex items-center gap-3">
                <span className="admin-text-body text-ink-secondary w-28 truncate font-mono">{type}</span>
                <div className="flex-1 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
                  <div className="h-full rounded-full bg-type-lesson/40" style={{ width: `${Math.max(pct, 2)}%` }} />
                </div>
                <span className="admin-text-body text-ink-faint font-mono tabular-nums w-8 text-right">{count}</span>
              </div>
            );
          })}
          {byType.length === 0 && <span className="admin-text-body text-ink-faint">No memories in period</span>}
        </div>
      </div>
      <div className="space-y-3">
        <div>
          <span className="admin-section-label">Totals</span>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
              <span className="admin-text-body text-ink-faint">All-time</span>
              <span className="admin-text-body text-ink font-mono tabular-nums">{totalAll.toLocaleString()}</span>
            </span>
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
              <span className="admin-text-body text-ink-faint">Period</span>
              <span className="admin-text-body text-ink font-mono tabular-nums">{periodTotal.toLocaleString()}</span>
            </span>
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
              <span className="admin-text-body text-ink-faint">Cloud</span>
              <span className="admin-text-body text-ink font-mono tabular-nums">{cloudCount.toLocaleString()}</span>
            </span>
          </div>
        </div>
        {insights.summary.growthPct !== -1 && (
          <div>
            <span className="admin-section-label">Growth vs Prior Period</span>
            <div className="mt-2">
              <span className={`admin-text-body font-mono tabular-nums ${insights.summary.growthPct >= 0 ? "text-type-lesson" : "text-type-error"}`}>
                {insights.summary.growthPct >= 0 ? "+" : ""}{insights.summary.growthPct.toFixed(1)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function KpiDetailSessions({ insights }: { insights: InsightsData }) {
  const alloc = insights.allocation ?? [];
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <span className="admin-section-label">Sessions by Project ({insights.period}d)</span>
        <div className="mt-2 space-y-1">
          {alloc.length > 0 ? alloc.slice(0, 8).map((p) => (
            <div key={p.displayName} className="flex items-center gap-3">
              <span className="admin-text-body text-ink-secondary w-32 truncate">{p.displayName}</span>
              <div className="flex-1 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
                <div className="h-full rounded-full bg-[#60a5fa]/40" style={{ width: `${Math.max(p.percent, 2)}%` }} />
              </div>
              <span className="admin-text-body text-ink-faint font-mono tabular-nums w-10 text-right">{p.percent.toFixed(0)}%</span>
            </div>
          )) : <span className="admin-text-body text-ink-faint">No session data</span>}
        </div>
      </div>
      <div>
        <span className="admin-section-label">Summary</span>
        <div className="mt-2 flex flex-wrap gap-2">
          <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
            <span className="admin-text-body text-ink-faint">Total Sessions</span>
            <span className="admin-text-body text-ink font-mono tabular-nums">{insights.summary.activeSessions}</span>
          </span>
          <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
            <span className="admin-text-body text-ink-faint">Projects</span>
            <span className="admin-text-body text-ink font-mono tabular-nums">{alloc.length}</span>
          </span>
        </div>
      </div>
    </div>
  );
}

function KpiDetailPublished({ insights }: { insights: InsightsData }) {
  if (!insights.content) return null;
  const tw = insights.content.tweets;
  const li = insights.content.linkedin;
  const rows = [
    { platform: "X / Twitter", ...tw },
    { platform: "LinkedIn", ...li },
  ];
  return (
    <div>
      <span className="admin-section-label">Content Pipeline ({insights.period}d)</span>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full admin-text-body">
          <thead>
            <tr className="text-ink-faint font-mono uppercase tracking-wider">
              <th className="text-left py-1.5 pr-4">Platform</th>
              <th className="text-right py-1.5 px-3">Published</th>
              <th className="text-right py-1.5 px-3">Pending</th>
              <th className="text-right py-1.5 px-3">Failed</th>
              <th className="text-right py-1.5 px-3">Retries</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.platform} className="border-t border-edge-subtle/30">
                <td className="py-2 pr-4 text-ink-secondary">{r.platform}</td>
                <td className="py-2 px-3 text-right font-mono tabular-nums text-type-lesson">{r.published}</td>
                <td className="py-2 px-3 text-right font-mono tabular-nums text-ink-faint">{r.pending}</td>
                <td className={`py-2 px-3 text-right font-mono tabular-nums ${r.failed > 0 ? "text-type-error" : "text-ink-faint"}`}>{r.failed}</td>
                <td className="py-2 px-3 text-right font-mono tabular-nums text-ink-faint">{r.retries ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <span className="admin-text-body text-ink-faint">Success Rate:</span>
        <span className={`admin-text-body font-mono tabular-nums font-medium ${insights.summary.publishSuccessRate >= 90 ? "text-type-lesson" : insights.summary.publishSuccessRate >= 70 ? "text-type-reminder" : "text-type-error"}`}>
          {insights.summary.publishSuccessRate.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function KpiDetailEngagement({ dashboard }: { dashboard: DashboardData }) {
  const perf = dashboard.performance;
  const overall = perf?.overall;
  const byType = perf?.byContentType?.sort((a, b) => b.avgEngagementRate - a.avgEngagementRate) ?? [];
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <span className="admin-section-label">By Content Type</span>
        <div className="mt-2 space-y-1">
          {byType.length > 0 ? byType.slice(0, 6).map((t) => (
            <div key={t.contentType} className="flex items-center gap-3">
              <span className="admin-text-body text-ink-secondary w-28 truncate">{t.contentType}</span>
              <div className="flex-1 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
                <div className="h-full rounded-full bg-type-lesson/40" style={{ width: `${Math.min(t.avgEngagementRate * 10, 100)}%` }} />
              </div>
              <span className="admin-text-body text-ink-faint font-mono tabular-nums w-12 text-right">{t.avgEngagementRate.toFixed(1)}%</span>
            </div>
          )) : <span className="admin-text-body text-ink-faint">No engagement data</span>}
        </div>
      </div>
      <div className="space-y-3">
        <div>
          <span className="admin-section-label">Overall</span>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
              <span className="admin-text-body text-ink-faint">Avg Rate</span>
              <span className="admin-text-body text-ink font-mono tabular-nums">{overall?.avgEngagementRate?.toFixed(1) ?? "0"}%</span>
            </span>
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
              <span className="admin-text-body text-ink-faint">Trend</span>
              <span className={`admin-text-body font-mono ${overall?.recentTrend === "improving" ? "text-type-lesson" : overall?.recentTrend === "declining" ? "text-type-error" : "text-ink-faint"}`}>
                {overall?.recentTrend ?? "steady"}
              </span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function KpiDetailDistribution({ dashboard }: { dashboard: DashboardData }) {
  const gh = dashboard.github;
  const pypi = dashboard.pypi;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <span className="admin-section-label">GitHub</span>
        <div className="mt-2 flex flex-wrap gap-2">
          {gh ? (
            <>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Stars</span>
                <span className="admin-text-body text-ink font-mono tabular-nums">{gh.stars}</span>
              </span>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Forks</span>
                <span className="admin-text-body text-ink font-mono tabular-nums">{gh.forks}</span>
              </span>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Issues</span>
                <span className="admin-text-body text-ink font-mono tabular-nums">{gh.openIssues}</span>
              </span>
            </>
          ) : <span className="admin-text-body text-ink-faint">Not connected</span>}
        </div>
      </div>
      <div>
        <span className="admin-section-label">PyPI</span>
        <div className="mt-2 flex flex-wrap gap-2">
          {pypi ? (
            <>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Weekly</span>
                <span className="admin-text-body text-ink font-mono tabular-nums">{fmtK(pypi.weeklyDownloads)}</span>
              </span>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Monthly</span>
                <span className="admin-text-body text-ink font-mono tabular-nums">{fmtK(pypi.monthlyDownloads)}</span>
              </span>
              <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle">
                <span className="admin-text-body text-ink-faint">Version</span>
                <span className="admin-text-body text-ink font-mono">{pypi.version}</span>
              </span>
            </>
          ) : <span className="admin-text-body text-ink-faint">Not connected</span>}
        </div>
      </div>
    </div>
  );
}

function KpiDetail({ label, insights, dashboard }: { label: string; insights: InsightsData; dashboard: DashboardData }) {
  switch (label) {
    case "Memories": return <KpiDetailMemories insights={insights} />;
    case "Sessions": return <KpiDetailSessions insights={insights} />;
    case "Published": return <KpiDetailPublished insights={insights} />;
    case "Engagement": return <KpiDetailEngagement dashboard={dashboard} />;
    case "Stars + Downloads": return <KpiDetailDistribution dashboard={dashboard} />;
    default: return null;
  }
}

// ─── Mini Sparkline ──────────────────────────────────────────

function MiniSparkline({ data, width = 56, height = 18, color = "rgba(94,201,160,0.5)" }: { data: number[]; width?: number; height?: number; color?: string }) {
  if (data.length < 3) return null;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={width} height={height} className="shrink-0 opacity-70">
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── KPI Strip ───────────────────────────────────────────────

function KpiStrip({ kpis, expanded, onToggle, insights, dashboard }: {
  kpis: KpiTile[];
  expanded: string | null;
  onToggle: (label: string) => void;
  insights: InsightsData;
  dashboard: DashboardData;
}) {
  return (
    <div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {kpis.map((kpi) => {
          const isOpen = expanded === kpi.label;
          return (
            <Tip key={kpi.label} text={kpi.tooltip}>
              <button
                onClick={() => onToggle(kpi.label)}
                className={`admin-card px-4 py-3.5 w-full text-left transition-colors ${
                  isOpen ? "border-gold/20 bg-surface-elevated/30" : "hover:border-edge-default"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="admin-text-metric text-ink">{kpi.value}</span>
                  {kpi.delta && (
                    <span className={`flex items-center gap-0.5 admin-text-delta px-1.5 py-0.5 rounded-full ${
                      kpi.trend === "up" ? "bg-type-lesson/10 text-type-lesson" :
                      kpi.trend === "down" ? "bg-type-error/10 text-type-error" :
                      "bg-surface-elevated text-ink-faint"
                    }`}>
                      <TrendIcon trend={kpi.trend} />
                      {kpi.delta}
                    </span>
                  )}
                </div>
                {kpi.sparkline && (
                  <div className="mt-1.5 mb-0.5">
                    <MiniSparkline
                      data={kpi.sparkline}
                      color={kpi.trend === "up" ? "rgba(94,201,160,0.5)" : kpi.trend === "down" ? "rgba(240,96,96,0.5)" : "rgba(152,152,176,0.3)"}
                    />
                  </div>
                )}
                <span className="admin-text-body text-ink-faint font-mono tracking-wide uppercase mt-1 block">{kpi.label}</span>
                {kpi.deltaContext && (
                  <span className="admin-text-micro text-ink-faint/50 font-mono mt-0.5 block">{kpi.deltaContext}</span>
                )}
              </button>
            </Tip>
          );
        })}
      </div>

      {/* Expandable detail panel */}
      <div className="grid transition-[grid-template-rows] duration-300 ease-out" style={{ gridTemplateRows: expanded ? "1fr" : "0fr" }}>
        <div className="overflow-hidden">
          {expanded && (
            <div className="admin-card mt-3 p-5">
              <KpiDetail label={expanded} insights={insights} dashboard={dashboard} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Problems Panel ──────────────────────────────────────────

function ProblemsPanel({ problems, onNavigate }: { problems: Problem[]; onNavigate: (tab: Tab) => void }) {
  if (problems.length === 0) return null;
  const urgent = problems.filter(p => p.urgent);
  const nonUrgent = problems.filter(p => !p.urgent);
  const [showNonUrgent, setShowNonUrgent] = useState(false);

  // Check snoozed alerts
  const snoozed = typeof window !== "undefined" ? JSON.parse(localStorage.getItem("admin_snoozed_alerts") ?? "{}") : {};
  const now = Date.now();

  const renderItem = (item: Problem, i: number) => {
    const mapping = mapProblemToAlert(item.text);
    if (mapping && snoozed[mapping.type] && snoozed[mapping.type] > now) return null;
    if (mapping) {
      return (
        <AlertCard
          key={`alert-${i}`}
          text={item.text}
          urgent={item.urgent}
          mapping={mapping}
          onNavigate={onNavigate}
        />
      );
    }
    return (
      <div key={`p-${i}`} className={`flex items-center gap-3 ${item.urgent ? "px-3 py-2 rounded-lg bg-gold/[0.03]" : "px-3 py-1.5"}`}>
        <span className={`rounded-full shrink-0 ${item.urgent ? "w-1.5 h-1.5 bg-gold/60" : "w-1 h-1 bg-ink-faint/20"}`} />
        <span className={`admin-text-body leading-snug flex-1 ${item.urgent ? "text-ink font-medium" : "text-ink-faint"}`}>{item.text}</span>
      </div>
    );
  };

  return (
    <div className="admin-card amber-border-glow p-5">
      <div className="flex items-center gap-2.5 mb-3">
        <span className="text-type-reminder"><AlertIcon /></span>
        <span className="admin-text-body font-mono font-medium text-ink-tertiary uppercase tracking-[0.08em]">Problems</span>
        {urgent.length > 0 && (
          <span className="admin-text-delta px-2 py-0.5 rounded-full bg-gold/10 text-gold border border-gold/[0.1]">
            {urgent.length} urgent
          </span>
        )}
      </div>

      {/* Urgent problems — always visible */}
      <div className="space-y-0.5">
        {urgent.map((item, i) => renderItem(item, i))}
      </div>

      {/* Non-urgent — collapsed by default */}
      {nonUrgent.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setShowNonUrgent(prev => !prev)}
            className="flex items-center gap-2 admin-text-caption text-ink-faint hover:text-ink-tertiary transition-colors"
          >
            <ChevronIcon open={showNonUrgent} />
            <span>{nonUrgent.length} suggestion{nonUrgent.length !== 1 ? "s" : ""}</span>
          </button>
          <div className="grid transition-[grid-template-rows] duration-200 ease-out" style={{ gridTemplateRows: showNonUrgent ? "1fr" : "0fr" }}>
            <div className="overflow-hidden">
              <div className="space-y-0.5 mt-2 pt-2 border-t border-edge-subtle/30">
                {nonUrgent.map((item, i) => renderItem(item, i + urgent.length))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Project Card Component ──────────────────────────────────

function ProjectCardView({ project, isOpen, onToggle }: { project: ProjectCard; isOpen: boolean; onToggle: () => void }) {
  const hasAttention = project.attention.some(a => a.urgent);
  const hasActive = project.active.length > 0;
  const ms = MOMENTUM_STYLE[project.momentum];

  return (
    <div className={`admin-card transition-all duration-200 ${hasAttention ? "amber-border-glow" : ""}`}>
      <button onClick={onToggle} className="w-full text-left p-5 cursor-pointer group">
        <div className="flex items-center gap-4">
          <div className="relative shrink-0">
            <GradeArc score={project.score} color={project.gradeColor} />
            <span className="absolute inset-0 flex items-center justify-center admin-text-body font-bold" style={{ color: project.gradeColor }}>{project.grade}</span>
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5">
              <span className="admin-text-body font-medium text-ink">{project.name}</span>
              {hasActive && (
                <span className="relative flex h-2 w-2 shrink-0">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-type-lesson opacity-30" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-type-lesson" />
                </span>
              )}
              <span className={`flex items-center gap-1 admin-text-delta px-2 py-0.5 rounded-full border ${ms.cls}`}>
                <TrendIcon trend={project.momentum === "up" ? "up" : project.momentum === "cooling" || project.momentum === "stalled" ? "down" : "flat"} />
                {project.momentumDelta || ms.label}
              </span>
            </div>
            <p className="admin-text-body text-ink-faint mt-0.5 truncate">
              {project.completed.length > 0 && `${project.completed.length} completed`}
              {project.active.length > 0 && ` · ${project.active.length} in progress`}
              {project.attention.length > 0 && ` · ${project.attention.length} need${project.attention.length === 1 ? "s" : ""} attention`}
              {project.sessionCount > 0 && ` · ${project.sessionCount} sessions`}
            </p>
          </div>

          {project.attention.filter(a => a.urgent).length > 0 && (
            <span className="px-2 py-0.5 rounded-full admin-text-delta bg-gold/10 text-gold/70 border border-gold/[0.12]">
              {project.attention.filter(a => a.urgent).length} urgent
            </span>
          )}

          <ChevronIcon open={isOpen} />
        </div>
      </button>

      <div className="grid transition-[grid-template-rows] duration-300 ease-out" style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}>
        <div className="overflow-hidden">
          <div className="px-5 pb-5 space-y-4">
            <div className="h-px bg-edge-subtle" />

            {/* Metrics */}
            {project.metrics.length > 0 && (
              <div>
                <span className="admin-section-label">Metrics</span>
                <div className="mt-2 flex flex-wrap gap-3">
                  {project.metrics.map((m, i) => {
                    const chip = (
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-elevated/50 border border-edge-subtle cursor-default">
                        <span className="admin-text-body text-ink-faint">{m.label}</span>
                        <span className="admin-text-body text-ink font-mono tabular-nums">{m.value}</span>
                        {m.trend && <TrendIcon trend={m.trend} />}
                      </div>
                    );
                    return m.tooltip ? <Tip key={i} text={m.tooltip}>{chip}</Tip> : <span key={i}>{chip}</span>;
                  })}
                </div>
              </div>
            )}

            {/* Recent decision */}
            {project.recentDecision && (
              <div className="flex items-start gap-2.5 px-3 py-2.5 rounded-lg bg-gold/[0.03] border border-gold/[0.06]">
                <span className="admin-text-body font-mono tracking-[0.1em] text-gold/40 uppercase shrink-0 mt-0.5">Decision</span>
                <span className="admin-text-body text-ink-secondary leading-snug">{project.recentDecision}</span>
              </div>
            )}

            {/* Completed — signal highlighted */}
            {project.completed.length > 0 && (
              <div>
                <span className="admin-section-label">Completed</span>
                <div className="mt-2 space-y-1.5">
                  {project.completed.filter(c => c.signal).map((item, i) => (
                    <div key={`s${i}`} className="flex items-start gap-3 px-3 py-2 rounded-lg bg-type-lesson/[0.03] border border-type-lesson/[0.06]">
                      <span className="mt-[3px] text-type-lesson/70 shrink-0"><CheckIcon /></span>
                      <span className="admin-text-body text-ink leading-snug flex-1 font-medium">{item.text}</span>
                      <span className="admin-text-body text-ink-faint font-mono tabular-nums shrink-0 mt-[2px]">{item.time}</span>
                    </div>
                  ))}
                  {project.completed.filter(c => !c.signal).map((item, i) => (
                    <div key={`n${i}`} className="flex items-start gap-3">
                      <span className="mt-[3px] text-type-lesson/30 shrink-0"><CheckIcon /></span>
                      <span className="admin-text-body text-ink-faint leading-snug flex-1">{item.text}</span>
                      <span className="admin-text-body text-ink-faint/50 font-mono tabular-nums shrink-0 mt-[2px]">{item.time}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* In Progress */}
            {project.active.length > 0 && (
              <div>
                <span className="admin-section-label">In Progress</span>
                <div className="mt-2 space-y-1.5">
                  {project.active.map((item, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span className="mt-[3px] text-[#60a5fa]/50 shrink-0"><SpinnerIcon /></span>
                      <span className="admin-text-body text-ink leading-snug flex-1">{item.text}</span>
                      {item.agent && (
                        <span className="admin-text-micro font-mono px-2 py-0.5 rounded-full bg-type-lesson/[0.08] text-type-lesson/60 border border-type-lesson/[0.1] shrink-0">{item.agent}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Needs Attention */}
            {project.attention.length > 0 && (
              <div>
                <span className="admin-section-label">Needs Attention</span>
                <div className="mt-2 space-y-1.5">
                  {project.attention.map((item, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span className={`mt-[3px] shrink-0 ${item.urgent ? "text-gold/60" : "text-ink-faint/30"}`}><AlertIcon /></span>
                      <span className={`admin-text-body leading-snug flex-1 ${item.urgent ? "text-ink font-medium" : "text-ink-secondary/60"}`}>{item.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Dashboard ──────────────────────────────────────────

export default function Dashboard({
  onNavigate,
  ambientStatus,
  account = "all",
  userId,
}: {
  onNavigate: (tab: Tab) => void;
  ambientStatus?: AmbientStatus;
  account?: "all" | "jasonsosa" | "omega_memory";
  userId?: string;
}) {
  const [insights, setInsights] = useState<InsightsData | null>(null);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [openProjects, setOpenProjects] = useState<Set<number>>(new Set());
  const [expandedKpi, setExpandedKpi] = useState<string | null>(null);
  const [layer, setLayer] = useState<1 | 2 | 3>(1);
  const { projects: registryProjects } = useProjects();

  const pausedIds = useMemo(() => {
    const ids = new Set<string>();
    for (const p of registryProjects) {
      if (p.is_paused) ids.add(p.id);
    }
    return ids;
  }, [registryProjects]);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [insRes, dashRes] = await Promise.all([
        fetch("/api/admin/insights?period=30"),
        fetch("/api/admin/dashboard"),
      ]);
      if (!insRes.ok) throw new Error(`Insights: ${insRes.status}`);
      if (!dashRes.ok) throw new Error(`Dashboard: ${dashRes.status}`);
      const [insData, dashData] = await Promise.all([insRes.json(), dashRes.json()]);
      setInsights(insData);
      setDashboard(dashData);
      setLastRefresh(new Date());
      _cache = { insights: insData, dashboard: dashData, ts: Date.now(), userId: userId ?? "" };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard");
    }
    setLoading(false);
  }, [userId]);

  useEffect(() => {
    if (_cache && _cache.userId === (userId ?? "")) {
      setInsights(_cache.insights);
      setDashboard(_cache.dashboard);
      setLastRefresh(new Date(_cache.ts));
      setLoading(false);
      if (Date.now() - _cache.ts > CACHE_TTL) fetchAll();
    } else {
      _cache = null;
      fetchAll();
    }
  }, [fetchAll, userId]);

  // Auto-escalate to L2 if there are urgent problems
  useEffect(() => {
    if (insights && dashboard) {
      const probs = buildProblems(insights, dashboard, pausedIds);
      if (probs.some(p => p.urgent)) {
        setLayer(2);
      }
    }
  }, [insights, dashboard, pausedIds]);

  // Loading skeleton
  if (loading && !insights) {
    return (
      <div className="px-6 sm:px-8 py-6 space-y-6">
        <div className="admin-card p-4 sm:p-5">
          <div className="flex items-center gap-3 sm:gap-5 flex-wrap">
            <div className="skeleton h-7 w-12 rounded-md" />
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="skeleton w-2.5 h-2.5 rounded-full" />
                <div className="skeleton h-5 w-16 rounded" />
              </div>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="admin-card p-4"><div className="skeleton h-8 w-16 rounded mb-2" /><div className="skeleton h-3 w-20 rounded" /></div>
          ))}
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="admin-card p-5"><div className="skeleton h-6 w-32 rounded mb-3" /><div className="skeleton h-4 w-48 rounded" /></div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !insights || !dashboard) {
    return (
      <div className="px-5 pt-6">
        <div className="p-5 bg-type-error/10 border border-type-error/20 rounded-xl text-sm text-type-error">
          {error || "No data available"}
        </div>
      </div>
    );
  }

  const kpis = buildKpis(insights, dashboard);
  const projects = buildProjects(insights, dashboard, pausedIds);
  const problems = buildProblems(insights, dashboard, pausedIds);

  // Count stalled/inactive radar projects for summary
  const stalledCount = insights.projects.filter(p => {
    const nameL = p.displayName.toLowerCase();
    if (nameL.includes("omega") || pausedIds.has(nameL)) return false;
    if (!p.isActive && p.daysSinceLastSession > 14) return false;
    return p.momentum === "stalled" || p.momentum === "cooling";
  }).length;

  const totalCompleted = projects.reduce((s, p) => s + p.completed.length, 0);
  const totalActive = projects.reduce((s, p) => s + p.active.length, 0);
  const totalUrgent = problems.filter(p => p.urgent).length;

  return (
    <div className="px-6 sm:px-8 py-6 space-y-6">
      {/* Summary bar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="admin-text-title text-ink">Today&apos;s Overview</h1>
          <p className="admin-text-body text-ink-faint mt-1">
            <span className="text-ink-secondary font-mono tabular-nums">{totalCompleted}</span> tasks completed
            {totalActive > 0 && <><span className="mx-1.5 text-ink-faint/30">·</span><span className="text-type-lesson font-mono tabular-nums">{totalActive}</span> in progress</>}
            {totalUrgent > 0 && <><span className="mx-1.5 text-ink-faint/30">·</span><span className="text-gold font-mono tabular-nums">{totalUrgent}</span> need attention</>}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchAll}
            disabled={loading}
            title={lastRefresh ? `Last refreshed: ${lastRefresh.toLocaleString()}` : undefined}
            className="flex items-center gap-2 px-4 py-2 rounded-lg admin-text-body font-medium text-ink-secondary hover:text-ink border border-edge hover:border-edge-default transition-colors disabled:opacity-50"
          >
            {loading ? <SpinnerIcon /> : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
              </svg>
            )}
            <span className="hidden sm:inline">Run Check</span>
          </button>
        </div>
      </div>

      {/* KPI Strip */}
      <KpiStrip
        kpis={kpis}
        expanded={expandedKpi}
        onToggle={(label) => setExpandedKpi(prev => prev === label ? null : label)}
        insights={insights}
        dashboard={dashboard}
      />

      {/* Coordination mini-widget (L1) */}
      {ambientStatus && ambientStatus.activeAgents > 0 && (
        <button
          onClick={() => onNavigate("coordination")}
          className="admin-card p-3.5 flex items-center gap-3 group hover:border-gold/20 transition-colors w-full text-left"
        >
          <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-type-lesson/[0.08] border border-type-lesson/[0.12] shrink-0">
            <svg className="w-4 h-4 text-type-lesson" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="admin-text-body font-medium text-ink-secondary">
                {ambientStatus.activeAgents} agent{ambientStatus.activeAgents !== 1 ? "s" : ""} active
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-type-lesson animate-pulse" />
            </div>
            <p className="admin-text-body text-ink-faint mt-0.5 truncate">
              {ambientStatus.totalSessions} session{ambientStatus.totalSessions !== 1 ? "s" : ""} total
              {ambientStatus.fileConflicts.length > 0 && (
                <span className="text-type-error ml-1.5">
                  {ambientStatus.fileConflicts.length} file conflict{ambientStatus.fileConflicts.length !== 1 ? "s" : ""}
                </span>
              )}
            </p>
          </div>
          <svg className="w-4 h-4 text-ink-faint group-hover:text-ink-tertiary transition-colors shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        </button>
      )}

      {/* Layer navigation */}
      <div className="flex items-center gap-2 px-1">
        {layer === 1 && problems.length > 0 && (
          <button
            onClick={() => setLayer(2)}
            className="admin-text-caption text-gold hover:text-gold/80 font-medium transition-colors flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
            </svg>
            {problems.length} problem{problems.length !== 1 ? "s" : ""} need attention
          </button>
        )}
        {layer > 1 && (
          <button
            onClick={() => setLayer(1)}
            className="text-[13px] text-ink-faint hover:text-ink-tertiary transition-colors"
          >
            Collapse
          </button>
        )}
        {layer === 2 && (
          <button
            onClick={() => setLayer(3)}
            className="ml-auto text-[13px] text-ink-faint hover:text-ink-tertiary transition-colors"
          >
            Raw data
          </button>
        )}
        {layer === 3 && (
          <button
            onClick={() => setLayer(2)}
            className="ml-auto text-[13px] text-ink-faint hover:text-ink-tertiary transition-colors"
          >
            Hide raw data
          </button>
        )}
      </div>

      {/* L2: Problems + Project grid */}
      {layer >= 2 && (
        <div className="space-y-6 card-enter">
          {/* Divider */}
          <div className="h-px bg-gradient-to-r from-gold/20 via-gold/10 to-transparent" />

          {/* Problems */}
          <ProblemsPanel problems={problems} onNavigate={onNavigate} />

          {/* Project grid */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {projects.map((project, i) => (
              <ProjectCardView
                key={project.name}
                project={project}
                isOpen={openProjects.has(i)}
                onToggle={() => setOpenProjects(prev => {
                  const next = new Set(prev);
                  if (next.has(i)) next.delete(i); else next.add(i);
                  return next;
                })}
              />
            ))}
          </div>

          {/* Stalled projects summary */}
          {stalledCount > 0 && (
            <button
              onClick={() => onNavigate("projects")}
              className="w-full text-left px-4 py-3 rounded-xl border border-edge-subtle/50 hover:border-edge-subtle transition-colors group"
            >
              <span className="admin-text-body text-ink-faint">
                {stalledCount} inactive project{stalledCount !== 1 ? "s" : ""} hidden
              </span>
              <span className="admin-text-body text-ink-tertiary ml-2 group-hover:text-ink-faint transition-colors">
                View in Projects &rarr;
              </span>
            </button>
          )}
        </div>
      )}

      {/* L3: Raw data */}
      {layer === 3 && (
        <DashboardRawData data={{ insights, dashboard } as Record<string, unknown>} />
      )}
    </div>
  );
}

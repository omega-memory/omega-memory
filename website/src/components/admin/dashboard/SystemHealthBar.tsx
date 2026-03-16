import type { InsightsData, DashboardData, Tab, HealthSubCheck } from "../lib/types";
import { timeAgo } from "../contentUtils";
import Tooltip from "../shared/Tooltip";

interface Props {
  insights: InsightsData;
  dashboard: DashboardData;
  lastRefresh: Date | null;
  loading: boolean;
  onRefresh: () => void;
  onNavigate: (tab: Tab) => void;
  pausedProjects?: Set<string>;
}

interface SystemCheck {
  label: string;
  status: "healthy" | "warning" | "error";
  detail?: string;
  tab?: Tab;
  tooltipTitle: string;
  tooltipDetails: HealthSubCheck[];
  lastRun?: string;
}

// ─── MCP health proxy ───────────────────────────────────────
// We can't call the MCP server from Vercel directly. Instead, we use
// memory write recency as a proxy: if OMEGA wrote a memory recently,
// the MCP server was running at that time.

function mcpStatus(lastMemoryAt?: string): { status: "healthy" | "warning" | "error"; label: string } {
  if (!lastMemoryAt) return { status: "error", label: "No data" };
  const ageMs = Date.now() - new Date(lastMemoryAt).getTime();
  const hours = ageMs / 3600000;
  if (hours < 24) return { status: "healthy", label: "Active" };
  if (hours < 72) return { status: "warning", label: "Stale" };
  return { status: "error", label: "Inactive" };
}

function deriveChecks(insights: InsightsData, dashboard: DashboardData): SystemCheck[] {
  const checks: SystemCheck[] = [];

  // ── OMEGA ──────────────────────────────────────────────────
  const mcp = mcpStatus(insights.lastMemoryAt);
  const memCount = insights.summary.totalMemories;
  const cloudCount = insights.memory.totalCloud ?? 0;

  const totalAll = insights.memory.totalAll;
  let omegaStatus: "healthy" | "warning" | "error" = mcp.status;
  if (totalAll === 0) omegaStatus = "error";

  checks.push({
    label: "OMEGA",
    status: omegaStatus,
    detail: `${totalAll.toLocaleString()} memories`,
    tooltipTitle: "OMEGA System Health",
    tooltipDetails: [
      { label: "Agent Server", value: mcp.label, status: mcp.status },
      { label: "Total Memories", value: totalAll.toLocaleString(), status: totalAll > 0 ? "healthy" : "error" },
      { label: `New (${insights.period}d)`, value: memCount.toLocaleString(), status: memCount > 0 ? "healthy" : "warning" },
      { label: "Cloud DB", value: cloudCount.toLocaleString(), status: cloudCount > 0 ? "healthy" : "warning" },
      { label: "Last Write", value: insights.lastMemoryAt ? timeAgo(insights.lastMemoryAt) : "Never" },
      { label: `Sessions (${insights.period}d)`, value: insights.summary.activeSessions.toLocaleString() },
    ],
    lastRun: insights.lastMemoryAt ? timeAgo(insights.lastMemoryAt) : undefined,
  });

  // ── Jobs ───────────────────────────────────────────────────
  const enabled = (insights.schedules ?? []).filter((s) => s.enabled);
  const failing = enabled.filter((s) => (s.lastStatus === "error" && s.lastRunAt) || s.overdue);
  const healthy = enabled.length - failing.length;
  const jobStatus: "healthy" | "warning" | "error" = failing.length > 0 ? "error" : "healthy";

  const jobTooltipDetails: HealthSubCheck[] = enabled.map((s) => ({
    label: s.label || s.name,
    value: s.lastRunAt ? timeAgo(s.lastRunAt) : "Never",
    status: (s.lastStatus === "error" && s.lastRunAt) ? "error" : s.overdue ? "warning" : "healthy",
  }));

  const lastJobRun = enabled
    .filter((s) => s.lastRunAt)
    .sort((a, b) => (b.lastRunAt! > a.lastRunAt! ? 1 : -1))[0];

  checks.push({
    label: `Jobs (${healthy}/${enabled.length})`,
    status: jobStatus,
    detail: failing.length > 0 ? failing.map((f) => f.name).join(", ") : undefined,
    tab: "jobs",
    tooltipTitle: "Scheduled Jobs",
    tooltipDetails: jobTooltipDetails.length > 0
      ? jobTooltipDetails
      : [{ label: "No jobs", value: "None enabled" }],
    lastRun: lastJobRun?.lastRunAt ? timeAgo(lastJobRun.lastRunAt) : undefined,
  });

  // ── Cloud ──────────────────────────────────────────────────
  // cloudCount is the all-time unfiltered Supabase count (same source as totalAll).
  // NOTE: We cannot distinguish "cloud DB is empty" from "cloud DB is unreachable"
  // because the dashboard API returns 0 for both cases. A proper health-check
  // endpoint would be needed to differentiate. For now, cloudCount > 0 means
  // "Connected", cloudCount === 0 means "Unknown" (could be empty or unreachable).
  let cloudStatus: "healthy" | "warning" | "error" = "healthy";
  if (!cloudCount) cloudStatus = "warning";

  const cloudConnectionLabel = cloudCount > 0 ? "Connected" : "Unknown";
  const cloudConnectionStatus: "healthy" | "warning" = cloudCount > 0 ? "healthy" : "warning";

  checks.push({
    label: "Cloud",
    status: cloudStatus,
    detail: cloudCount
      ? `${cloudCount.toLocaleString()} stored`
      : "No cloud data",
    tooltipTitle: "Cloud Database",
    tooltipDetails: [
      { label: "Total", value: cloudCount.toLocaleString(), status: cloudCount > 0 ? "healthy" : "warning" },
      { label: `New (${insights.period}d)`, value: memCount.toLocaleString() },
      { label: "Connection", value: cloudConnectionLabel, status: cloudConnectionStatus },
    ],
  });

  // ── X ──────────────────────────────────────────────────────
  const tw = dashboard.twitter;
  const xAccts = tw?.accounts;
  const totalFollowers = xAccts
    ? (xAccts.jasonsosa?.followers ?? 0) + (xAccts.omega_memory?.followers ?? 0)
    : tw?.followers ?? 0;

  const xTooltipDetails: HealthSubCheck[] = tw
    ? [
        { label: "Connection", value: "Connected", status: "healthy" },
        ...(xAccts?.jasonsosa
          ? [{ label: "@jasonsosa", value: xAccts.jasonsosa.followers.toLocaleString() }]
          : []),
        ...(xAccts?.omega_memory
          ? [{ label: "@omega_memory", value: xAccts.omega_memory.followers.toLocaleString() }]
          : []),
        ...(!xAccts ? [{ label: "Followers", value: tw.followers.toLocaleString() }] : []),
        { label: "Total", value: totalFollowers.toLocaleString() },
        { label: "Tweets", value: tw.tweetCount.toLocaleString() },
      ]
    : [{ label: "Connection", value: "Disconnected", status: "error" as const }];

  checks.push({
    label: "X",
    status: tw ? "healthy" : "warning",
    detail: tw ? `${totalFollowers.toLocaleString()} followers` : "Disconnected",
    tooltipTitle: "X (Twitter) Integration",
    tooltipDetails: xTooltipDetails,
  });

  return checks;
}

// ─── Status dot styles ──────────────────────────────────────

const STATUS_DOT: Record<string, string> = {
  healthy: "bg-type-lesson shadow-[0_0_6px_rgba(94,201,160,0.4)]",
  warning: "bg-type-reminder shadow-[0_0_6px_rgba(245,158,11,0.4)]",
  error:   "bg-type-error shadow-[0_0_6px_rgba(240,96,96,0.4)]",
};

const STATUS_DOT_SM: Record<string, string> = {
  healthy: "bg-type-lesson",
  warning: "bg-type-reminder",
  error:   "bg-type-error",
};

// ─── Tooltip content renderer ───────────────────────────────

function TooltipContent({ title, details }: { title: string; details: HealthSubCheck[] }) {
  return (
    <div className="space-y-1.5">
      <div className="font-semibold text-ink admin-text-caption pb-1 border-b border-edge-subtle mb-1.5">
        {title}
      </div>
      {details.map((d) => (
        <div key={d.label} className="flex items-center justify-between gap-3 admin-text-caption">
          <span className="flex items-center gap-1.5 text-ink-secondary">
            {d.status && (
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${STATUS_DOT_SM[d.status]}`} />
            )}
            {d.label}
          </span>
          <span className="text-ink font-medium tabular-nums">{d.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main component ─────────────────────────────────────────

export default function SystemHealthBar({
  insights,
  dashboard,
  lastRefresh,
  loading,
  onRefresh,
  onNavigate,
}: Props) {
  const checks = deriveChecks(insights, dashboard);

  return (
    <div className="admin-toolbar">
      {/* Top row: indicators + timestamp + button */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 sm:gap-5 flex-wrap">
          {/* Indicator dots with tooltips */}
          {checks.map((c) => {
            const Wrapper = c.tab ? "button" : "span";
            const indicator = (
              <Wrapper
                className={`flex items-center gap-2 admin-text-body ${
                  c.tab
                    ? "hover:text-ink transition-colors cursor-pointer"
                    : ""
                } ${
                  c.status === "healthy"
                    ? "text-ink-secondary"
                    : c.status === "warning"
                      ? "text-type-reminder"
                      : "text-type-error"
                }`}
                {...(c.tab ? { onClick: () => onNavigate(c.tab!) } : {})}
              >
                <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${STATUS_DOT[c.status]}`} />
                <span className="font-medium">{c.label}</span>
              </Wrapper>
            );

            return (
              <Tooltip
                key={c.label}
                content={<TooltipContent title={c.tooltipTitle} details={c.tooltipDetails} />}
                side="bottom"
                align="center"
              >
                {indicator}
              </Tooltip>
            );
          })}
        </div>

        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span
              className="admin-text-caption text-ink-tertiary font-mono tabular-nums"
              title={lastRefresh.toLocaleString()}
            >
              {timeAgo(lastRefresh.toISOString())}
            </span>
          )}
          <button
            onClick={onRefresh}
            disabled={loading}
            title="Run health check on all systems"
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg admin-text-body font-medium text-ink-secondary hover:text-ink bg-surface-elevated hover:bg-surface-hover border border-edge transition-colors disabled:opacity-50"
          >
            {loading ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182M2.985 19.644l3.181-3.182" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
              </svg>
            )}
            <span className="hidden sm:inline">Run Check</span>
          </button>
        </div>
      </div>
    </div>
  );
}

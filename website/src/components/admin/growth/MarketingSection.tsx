import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// ═══════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════

interface CalendarSlot {
  id: string;
  slot_date: string;
  slot_type: string;
  x_account: string;
  status: string;
  angle: string | null;
  draft_content: string | null;
}

interface Metrics {
  streak: number;
  this_week: { published: number; target: number; on_track: boolean };
  last_30_days: { total_slots: number; published: number; skipped: number; draft_ready: number };
  by_type: Record<string, { count: number; published: number; skipped: number }>;
}

interface Relationship {
  id: string;
  handle: string;
  display_name: string | null;
  tier: number;
  last_engaged_at: string | null;
  topics: string[] | null;
}

interface AgentHealth {
  [label: string]: string[];
}

interface GrowthLeaderboard {
  dimension: string;
  value: string;
  win_probability: number;
  sample_count: number;
}

interface FunnelData {
  impressions: number;
  engagements: number;
  link_clicks: number;
  site_visits: number;
  repo_visits: number;
  stars: number;
  downloads: number;
}

interface Snapshot {
  id: string;
  created_at: string;
  star_count: number | null;
  weekly_downloads: number | null;
  avg_funnel_score: number | null;
  experiment_count: number;
  snapshot: unknown;
}

// ═══════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════

const TYPE_LABELS: Record<string, string> = {
  build_log: "Build log",
  tweet: "Tweet",
  engagement: "Engagement",
  proof_point: "Proof point",
  thread: "Thread",
};

const PIPELINE_STAGES = [
  { key: "pending", label: "Writing" },
  { key: "draft_ready", label: "Review" },
  { key: "approved", label: "Ready" },
  { key: "published", label: "Posted" },
] as const;

const AUTOMATION_AGENTS = [
  "com.omega.github.seed-calendar",
  "com.omega.github.generate-briefs",
  "com.omega.vercel.publish",
  "com.omega.github.generate-hourly",
];

// Translate internal codenames to CMO-readable labels
const WEDGE_LABELS: Record<string, string> = {
  coordination: "Multi-agent coordination",
  "cross-client": "Works across all editors",
  "memory-quality": "Memory quality and depth",
  "vs-claude-mem": "Better than claude-mem",
  "agent-teams-fix": "Fixing agent team issues",
};
const FORMAT_LABELS: Record<string, string> = {
  "demo-screenshot": "Demo with screenshot",
  thread: "Thread",
  "single-insight": "Single insight",
  "code-snippet": "Code snippet",
  comparison: "Comparison",
};
const CTA_LABELS: Record<string, string> = {
  "repo-link": "Link to repo",
  "blog-link": "Link to blog",
  "no-cta": "No call to action",
  "try-it-command": "Try-it command",
};

function friendlyLabel(dimension: string, value: string): string {
  if (dimension === "wedge") return WEDGE_LABELS[value] || value;
  if (dimension === "format") return FORMAT_LABELS[value] || value;
  if (dimension === "cta") return CTA_LABELS[value] || value;
  return value.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ═══════════════════════════════════════════════════════════
// Review card
// ═══════════════════════════════════════════════════════════

function ReviewCard({
  slot,
  onUpdate,
  updating,
}: {
  slot: CalendarSlot;
  onUpdate: (id: string, updates: Record<string, unknown>) => void;
  updating: string | null;
}) {
  const [draft, setDraft] = useState(slot.draft_content || "");
  const [editing, setEditing] = useState(false);
  const isUpdating = updating === slot.id;
  const isDirty = draft !== (slot.draft_content || "");
  const isShort = slot.slot_type === "tweet" || slot.slot_type === "proof_point";
  const dayLabel = new Date(slot.slot_date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });

  const handleSave = () => {
    onUpdate(slot.id, { draft_content: draft, status: draft.trim() ? "draft_ready" : slot.status });
    setEditing(false);
  };

  return (
    <div className="rounded-xl border border-edge/30 bg-surface/40 overflow-hidden">
      <div className="px-4 py-3 flex items-center gap-3 border-b border-edge/15">
        <span className="text-[14px] font-medium text-ink">{TYPE_LABELS[slot.slot_type] || slot.slot_type}</span>
        <span className="text-[13px] text-ink-faint">@{slot.x_account}</span>
        <span className="text-[13px] text-ink-faint ml-auto">{dayLabel}</span>
      </div>
      <div className="p-4 space-y-3">
        {slot.angle && !editing && (
          <p className="text-[13px] text-ink-tertiary italic">{slot.angle}</p>
        )}
        {slot.draft_content && !editing ? (
          <div className="text-[15px] text-ink leading-relaxed whitespace-pre-wrap">{slot.draft_content}</div>
        ) : editing ? (
          <div>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={isShort ? 3 : 5}
              className="w-full bg-surface-elevated border border-edge/50 rounded-lg px-3 py-2.5 text-[15px] text-ink leading-relaxed placeholder:text-ink-faint/30 focus:outline-none focus:ring-1 focus:ring-gold/30 resize-y"
              placeholder="Write your draft..."
            />
            {isShort && (
              <div className={`text-[12px] text-right mt-1 ${draft.length > 280 ? "text-red-400" : "text-ink-faint"}`}>
                {draft.length}/280
              </div>
            )}
          </div>
        ) : (
          <p className="text-[14px] text-ink-faint italic">No draft yet</p>
        )}
        <div className="flex items-center gap-2 pt-1">
          {editing && isDirty ? (
            <>
              <button onClick={handleSave} disabled={isUpdating} className="px-4 py-2 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/20 hover:bg-gold/20 transition-colors disabled:opacity-50 cursor-pointer min-h-[44px]">
                {isUpdating ? "Saving..." : "Save"}
              </button>
              <button onClick={() => { setDraft(slot.draft_content || ""); setEditing(false); }} className="px-4 py-2 rounded-lg text-[14px] font-medium text-ink-faint border border-edge/30 hover:bg-surface-elevated transition-colors cursor-pointer min-h-[44px]">
                Cancel
              </button>
            </>
          ) : (
            <>
              {slot.status === "draft_ready" && (
                <button onClick={() => onUpdate(slot.id, { status: "approved" })} disabled={isUpdating} className="px-4 py-2 rounded-lg text-[14px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors disabled:opacity-50 cursor-pointer min-h-[44px]">
                  Approve
                </button>
              )}
              <button onClick={() => setEditing(true)} className="px-4 py-2 rounded-lg text-[14px] font-medium text-ink-secondary border border-edge/30 hover:bg-surface-elevated transition-colors cursor-pointer min-h-[44px]">
                Edit
              </button>
              {slot.status !== "skipped" && slot.status !== "published" && (
                <button onClick={() => onUpdate(slot.id, { status: "skipped" })} disabled={isUpdating} className="px-3 py-2 rounded-lg text-[14px] text-ink-faint hover:bg-surface-elevated transition-colors disabled:opacity-50 cursor-pointer min-h-[44px]">
                  Skip
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// Main component — unified Growth dashboard
// ═══════════════════════════════════════════════════════════

export default function MarketingSection() {
  // Content state
  const [slots, setSlots] = useState<CalendarSlot[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [agentHealth, setAgentHealth] = useState<AgentHealth>({});
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Performance state
  const [leaderboard, setLeaderboard] = useState<GrowthLeaderboard[]>([]);
  const [funnelThis, setFunnelThis] = useState<FunnelData | null>(null);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [chartRange, setChartRange] = useState<7 | 30>(7);

  // ─── Fetch all data in one layer ──────────────────────
  const fetchData = useCallback(async () => {
    try {
      setActionError(null);
      const now = new Date();
      const weekStart = new Date(now);
      weekStart.setDate(weekStart.getDate() - weekStart.getDay());

      const [calRes, metRes, relRes, healthRes, lbRes, f7Res] = await Promise.all([
        fetch(`/api/marketing/calendar?week=${weekStart.toISOString().slice(0, 10)}`),
        fetch("/api/marketing/metrics"),
        fetch("/api/marketing/relationships?stale=14"),
        fetch("/api/admin/schedule-runs/health"),
        fetch("/api/growth?type=leaderboard"),
        fetch("/api/growth?type=funnel&days=7"),
      ]);

      if (calRes.ok) setSlots((await calRes.json()).slots || []);
      if (metRes.ok) setMetrics(await metRes.json());
      if (relRes.ok) setRelationships((await relRes.json()).relationships || []);
      if (healthRes.ok) setAgentHealth((await healthRes.json()).health || {});
      if (lbRes.ok) {
        const data = await lbRes.json();
        setLeaderboard(data.leaderboard || []);
      }
      if (f7Res.ok) setFunnelThis((await f7Res.json()).funnel || null);
    } catch (err) {
      console.error("Growth fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  // Snapshots fetch once (updates every ~12h)
  useEffect(() => {
    fetch("/api/growth?type=snapshots")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.snapshots) setSnapshots(d.snapshots); })
      .catch(() => {});
  }, []);

  // ─── Actions ──────────────────────────────────────────
  const handleUpdateSlot = async (id: string, updates: Record<string, unknown>) => {
    try {
      setUpdating(id);
      const res = await fetch("/api/marketing/calendar", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, ...updates }),
      });
      if (res.ok) {
        const { slot } = await res.json();
        setSlots((prev) => prev.map((s) => (s.id === id ? { ...s, ...slot } : s)));
      }
    } catch (err) {
      setActionError("Failed to update slot. Try again.");
    } finally {
      setUpdating(null);
    }
  };

  const handleGenerateDrafts = async () => {
    setGenerating(true);
    try {
      const res = await fetch("/api/marketing/generate", { method: "POST" });
      if (res.ok) await fetchData();
    } catch (err) {
      setActionError("Failed to generate drafts. Try again.");
    } finally {
      setGenerating(false);
    }
  };

  // ─── Derived data ─────────────────────────────────────
  const reviewQueue = slots.filter((s) => s.status === "draft_ready");
  const pendingSlots = slots.filter((s) => s.status === "pending");
  const approvedSlots = slots.filter((s) => s.status === "approved");
  const publishedSlots = slots.filter((s) => s.status === "published");
  const total = slots.length;
  const posted = publishedSlots.length;
  const streak = metrics?.streak ?? 0;

  // Automation: are all agents healthy?
  const allStatuses = AUTOMATION_AGENTS.flatMap((key) => agentHealth[key] || []);
  const hasErrors = allStatuses.some((s) => s === "error");
  const hasRuns = allStatuses.length > 0;
  const automationOk = hasRuns && !hasErrors;

  // Best performing content
  const topTopic = leaderboard
    .filter((e) => e.dimension === "wedge")
    .sort((a, b) => b.win_probability - a.win_probability)[0];
  const topFormat = leaderboard
    .filter((e) => e.dimension === "format")
    .sort((a, b) => b.win_probability - a.win_probability)[0];

  // Pipeline counts
  const stageCounts = PIPELINE_STAGES.map((stage) => ({
    ...stage,
    count: slots.filter((s) => s.status === stage.key).length,
  }));

  // Snapshot-based metrics (actual totals, matches the chart)
  const snapshotMetrics = useMemo(() => {
    if (!snapshots.length) return null;
    const sorted = [...snapshots].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    const latest = sorted[0];

    // Find a snapshot from ~7 days ago for WoW comparison
    const weekAgo = Date.now() - 7 * 86_400_000;
    const prior = sorted.find((s) => new Date(s.created_at).getTime() <= weekAgo) || sorted[sorted.length - 1];

    const wow = (current: number | null, previous: number | null) => {
      const c = current ?? 0;
      const p = previous ?? 0;
      if (p === 0) return { value: c, pct: 0, dir: "flat" as const };
      const pct = Math.round(((c - p) / p) * 100);
      return { value: c, pct, dir: (pct > 0 ? "up" : pct < 0 ? "down" : "flat") as "up" | "down" | "flat" };
    };

    return {
      stars: wow(latest.star_count, prior.star_count),
      downloads: wow(latest.weekly_downloads, prior.weekly_downloads),
    };
  }, [snapshots]);

  // Experiment-scoped funnel metrics (impressions/engagements from tracked posts only)
  const experimentMetrics = useMemo(() => {
    if (!funnelThis) return null;
    return {
      impressions: funnelThis.impressions,
      engagements: funnelThis.engagements,
    };
  }, [funnelThis]);

  // Chart data
  const chartData = useMemo(() => {
    const cutoff = chartRange === 7 ? Date.now() - 7 * 86_400_000 : Date.now() - 30 * 86_400_000;
    return snapshots
      .filter((s) => new Date(s.created_at).getTime() >= cutoff && (s.star_count != null || s.weekly_downloads != null))
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .map((s) => ({ date: shortDate(s.created_at), stars: s.star_count ?? undefined, downloads: s.weekly_downloads ?? undefined }));
  }, [snapshots, chartRange]);

  // ─── Loading ──────────────────────────────────────────
  if (loading) {
    return (
      <div className="px-5 pt-6 pb-12 space-y-6">
        <div className="h-7 w-56 rounded-lg bg-surface/20 animate-pulse" />
        <div className="h-14 w-full rounded-xl bg-surface/20 animate-pulse" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => <div key={i} className="h-20 rounded-xl bg-surface/20 animate-pulse" />)}
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 pt-6 pb-12 space-y-8">

      {/* ════════════════════════════════════════════════════ */}
      {/* ZONE 1: Status + Action                            */}
      {/* ════════════════════════════════════════════════════ */}

      <section className="space-y-5">
        {/* Status bar */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-[18px] font-medium text-ink">Content</h2>
            <p className="text-[14px] text-ink-tertiary mt-0.5">
              {reviewQueue.length > 0
                ? `${reviewQueue.length} draft${reviewQueue.length > 1 ? "s" : ""} ready for your review`
                : total > 0 && posted === total
                  ? "All content posted this week"
                  : total > 0
                    ? `${posted} of ${total} posted this week`
                    : "No content scheduled"
              }
              {streak > 0 && <span className="text-gold"> · {streak}-week streak</span>}
              {hasRuns && (
                <span className="inline-flex items-center gap-1.5 ml-2">
                  <span className={`w-1.5 h-1.5 rounded-full inline-block ${automationOk ? "bg-emerald-400" : "bg-red-400"}`} />
                  <span className={automationOk ? "text-ink-faint" : "text-red-400"}>
                    {automationOk ? "Automation healthy" : "Automation issues"}
                  </span>
                </span>
              )}
            </p>
          </div>
          {pendingSlots.length > 0 && (
            <button
              onClick={handleGenerateDrafts}
              disabled={generating}
              className="px-4 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/20 hover:bg-gold/20 transition-colors disabled:opacity-50 cursor-pointer min-h-[44px] shrink-0 flex items-center gap-2"
            >
              {generating ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Writing...
                </>
              ) : "Write drafts"}
            </button>
          )}
        </div>

        {/* Pipeline strip */}
        {total > 0 && (
          <div className="flex items-center gap-px rounded-xl overflow-hidden border border-edge/20">
            {stageCounts.map((stage, i) => {
              const isActive = stage.count > 0;
              return (
                <div key={stage.key} className={`flex-1 flex items-center justify-center gap-2 py-3 px-2 ${isActive ? "bg-surface/50" : "bg-surface/20"}`}>
                  <span className="text-[13px] text-ink-secondary">{stage.label}</span>
                  <span className={`text-[16px] font-medium tabular-nums ${isActive ? "text-ink" : "text-ink-faint"}`}>{stage.count}</span>
                  {i < stageCounts.length - 1 && (
                    <svg className="w-3 h-3 text-ink-faint/20 ml-auto shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                    </svg>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {actionError && (
          <div className="p-3 rounded-xl text-[14px] text-type-error bg-type-error/[0.06] border border-type-error/[0.1] flex items-center justify-between">
            <span>{actionError}</span>
            <button onClick={() => setActionError(null)} className="text-type-error/60 hover:text-type-error ml-2 shrink-0">&times;</button>
          </div>
        )}

        {/* Review queue */}
        {reviewQueue.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-[14px] font-medium text-ink-secondary flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              Ready for review
            </h3>
            {reviewQueue.map((slot) => (
              <ReviewCard key={slot.id} slot={slot} onUpdate={handleUpdateSlot} updating={updating} />
            ))}
          </div>
        )}

        {/* Empty state */}
        {total === 0 && (
          <div className="rounded-xl border border-edge/20 bg-surface/20 p-8 text-center">
            <p className="text-[16px] text-ink-tertiary font-medium">No content scheduled this week</p>
            <p className="text-[14px] text-ink-faint mt-1">Content is automatically scheduled every Sunday</p>
          </div>
        )}
      </section>

      {/* ════════════════════════════════════════════════════ */}
      {/* ZONE 2: Performance                                 */}
      {/* ════════════════════════════════════════════════════ */}

      <section className="space-y-5">
        <div className="flex items-baseline justify-between">
          <h2 className="text-[18px] font-medium text-ink">Performance</h2>
          <span className="text-[13px] text-ink-faint">Actual totals · all accounts</span>
        </div>

        {/* Metric cards — snapshot-based actuals */}
        {snapshotMetrics && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {([
              ["Stars", "GitHub total", snapshotMetrics.stars],
              ["Downloads", "PyPI weekly", snapshotMetrics.downloads],
              ...(experimentMetrics ? [
                ["Impressions", "Tracked posts, 7d", { value: experimentMetrics.impressions, pct: 0, dir: "flat" as const }],
                ["Engagements", "Tracked posts, 7d", { value: experimentMetrics.engagements, pct: 0, dir: "flat" as const }],
              ] : []),
            ] as [string, string, { value: number; pct: number; dir: "up" | "down" | "flat" }][]).map(([label, subtitle, m]) => (
              <div key={label} className="rounded-xl border border-edge/20 bg-surface/20 p-4">
                <div className="text-[13px] text-ink-tertiary mb-0.5">{label}</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-[20px] font-medium tabular-nums text-ink">{m.value.toLocaleString()}</span>
                  {m.dir === "up" && <span className="text-[13px] font-medium text-emerald-400">+{m.pct}%</span>}
                  {m.dir === "down" && <span className="text-[13px] text-ink-faint">{m.pct}%</span>}
                </div>
                <div className="text-[12px] text-ink-faint mt-1">{subtitle}</div>
              </div>
            ))}
          </div>
        )}

        {/* Trend chart */}
        <div className="rounded-xl border border-edge/20 bg-surface/20 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[14px] font-medium text-ink-secondary">Stars and downloads over time</h3>
            <div className="flex gap-1">
              {([7, 30] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setChartRange(d)}
                  className={`px-2.5 py-1.5 text-[12px] rounded-md font-medium transition-colors cursor-pointer ${
                    chartRange === d ? "bg-gold/[0.12] text-gold" : "text-ink-tertiary hover:text-ink-secondary"
                  }`}
                >
                  {d === 7 ? "7 days" : "30 days"}
                </button>
              ))}
            </div>
          </div>
          {chartData.length >= 2 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "var(--color-ink-tertiary)", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={60}
                />
                <YAxis tick={{ fill: "var(--color-ink-tertiary)", fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
                <Tooltip contentStyle={{ backgroundColor: "var(--color-surface-elevated)", border: "1px solid var(--color-edge)", borderRadius: 8, fontSize: 12, color: "var(--color-ink)" }} />
                <Line type="monotone" dataKey="stars" name="Stars" stroke="#d4a843" strokeWidth={2} dot={false} connectNulls />
                <Line type="monotone" dataKey="downloads" name="Downloads" stroke="#60a5fa" strokeWidth={2} dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[200px] text-[14px] text-ink-tertiary">
              Not enough data for trends yet
            </div>
          )}
        </div>

        {/* What's working */}
        {(topTopic || topFormat) && (
          <div className="rounded-xl border border-edge/20 bg-surface/20 p-4 space-y-2">
            <h3 className="text-[14px] font-medium text-ink-secondary">What's working</h3>
            {topTopic && (
              <p className="text-[14px] text-ink-tertiary leading-relaxed">
                <span className="text-ink">Best topic:</span> {friendlyLabel("wedge", topTopic.value)}
                {topTopic.sample_count >= 3 && (
                  <span className="text-ink-faint"> -- outperforms other topics across {topTopic.sample_count} posts</span>
                )}
              </p>
            )}
            {topFormat && (
              <p className="text-[14px] text-ink-tertiary leading-relaxed">
                <span className="text-ink">Best format:</span> {friendlyLabel("format", topFormat.value)}
                {topFormat.sample_count >= 3 && (
                  <span className="text-ink-faint"> -- strongest engagement across {topFormat.sample_count} posts</span>
                )}
              </p>
            )}
          </div>
        )}
      </section>

      {/* ════════════════════════════════════════════════════ */}
      {/* ZONE 3: Context                                     */}
      {/* ════════════════════════════════════════════════════ */}

      {(approvedSlots.length > 0 || pendingSlots.length > 0 || publishedSlots.length > 0 || relationships.length > 0) && (
        <section className="space-y-4">
          {/* Approved — ready to go live */}
          {approvedSlots.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[14px] font-medium text-ink-secondary">Ready to post</h3>
              {approvedSlots.map((slot) => (
                <div key={slot.id} className="rounded-lg border border-edge/20 bg-surface/20 px-4 py-3">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[14px] text-ink font-medium">{TYPE_LABELS[slot.slot_type] || slot.slot_type}</span>
                    <span className="text-[13px] text-ink-faint">@{slot.x_account}</span>
                  </div>
                  {slot.draft_content && (
                    <p className="text-[14px] text-ink-tertiary leading-relaxed line-clamp-2">{slot.draft_content}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Pending — waiting for AI */}
          {pendingSlots.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[14px] font-medium text-ink-secondary">Waiting for drafts</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {pendingSlots.map((slot) => {
                  const dayLabel = new Date(slot.slot_date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
                  return (
                    <div key={slot.id} className="rounded-lg border border-edge/20 bg-surface/20 p-3">
                      <div className="text-[14px] text-ink">{TYPE_LABELS[slot.slot_type] || slot.slot_type}</div>
                      <div className="text-[13px] text-ink-faint">{dayLabel} · @{slot.x_account}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Posted this week */}
          {publishedSlots.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[14px] font-medium text-ink-secondary">Posted this week</h3>
              <div className="flex flex-wrap gap-2">
                {publishedSlots.map((slot) => {
                  const dayLabel = new Date(slot.slot_date + "T12:00:00").toLocaleDateString("en-US", { weekday: "short" });
                  return (
                    <div key={slot.id} className="rounded-lg border border-edge/15 bg-surface/20 px-3 py-2 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span className="text-[13px] text-ink-tertiary">{dayLabel}</span>
                      <span className="text-[13px] text-ink-secondary">{TYPE_LABELS[slot.slot_type] || slot.slot_type}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Reconnect */}
          {relationships.length > 0 && (
            <div className="rounded-xl border border-edge/20 bg-surface/20 p-4">
              <h3 className="text-[14px] font-medium text-ink-secondary mb-2">People to reconnect with</h3>
              <p className="text-[14px] text-ink-tertiary leading-relaxed">
                {relationships.slice(0, 4).map((rel, i) => {
                  const days = rel.last_engaged_at
                    ? Math.floor((Date.now() - new Date(rel.last_engaged_at).getTime()) / 86400000)
                    : null;
                  return (
                    <span key={rel.id}>
                      {i > 0 && ", "}
                      <span className="text-ink-secondary">@{rel.handle}</span>
                      {days !== null && <span className="text-ink-faint"> ({days} days ago)</span>}
                    </span>
                  );
                })}
                {relationships.length > 4 && (
                  <span className="text-ink-faint"> and {relationships.length - 4} more</span>
                )}
              </p>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

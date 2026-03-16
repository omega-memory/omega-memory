import { useEffect, useState } from "react";

interface AnalyticsSummary {
  totalPublished: number;
  totalPending: number;
  totalRejected: number;
  publishedThisWeek: number;
  repliesSentThisWeek: number;
  topTweets: {
    id: string;
    text: string;
    content_type: string;
    published_at: string;
    x_post_url: string | null;
    length_category: string;
  }[];
  byContentType: {
    content_type: string;
    count: number;
    published: number;
  }[];
  engagement: {
    replies_sent: number;
    replies_pending: number;
    targets_tracked: number;
    alerts_pending: number;
  };
  dailyCounts: { date: string; count: number }[];
}

interface PerformanceInsights {
  byContentType: { contentType: string; avgEngagementRate: number; count: number }[];
  byTheme: { theme: string; avgEngagementRate: number; count: number }[];
  bySlot: { slotNumber: number; avgEngagementRate: number; count: number }[];
  topPerformers: {
    text: string;
    contentType: string;
    engagementRate: number;
    impressions: number;
    likes: number;
    retweets: number;
    replies: number;
    bookmarks: number;
  }[];
  overall: {
    totalTracked: number;
    avgEngagementRate: number;
    recentTrend: "improving" | "steady" | "declining";
  };
}

const CONTENT_TYPE_LABELS: Record<string, string> = {
  insight: "Insight",
  conversation: "Conversation",
  build_log: "Build Log",
  proof_point: "Proof Point",
  manual: "Manual",
};

const TREND_ICON: Record<string, string> = {
  improving: "\u2191",
  declining: "\u2193",
  steady: "\u2192",
};

type AccountFilter = "all" | "jasonsosa" | "omega_memory";

export default function SocialAnalytics({ account = "all" }: { account?: AccountFilter }) {
  const [data, setData] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [perf, setPerf] = useState<PerformanceInsights | null>(null);
  const [perfLoading, setPerfLoading] = useState(false);
  const [perfError, setPerfError] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/analytics/summary");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    async function loadPerf() {
      setPerfLoading(true);
      setPerfError(false);
      try {
        const params = new URLSearchParams();
        if (account !== "all") params.set("account", account);
        const res = await fetch(`/api/admin/analytics/performance?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setPerf(await res.json());
      } catch {
        setPerfError(true);
      } finally {
        setPerfLoading(false);
      }
    }
    loadPerf();
  }, [account]);

  if (loading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-20 rounded-xl bg-surface-elevated skeleton" />
          ))}
        </div>
        <div className="h-32 rounded-xl bg-surface-elevated skeleton" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-ink-faint text-[15px]">
          {error ? `Failed to load analytics: ${error}` : "Unable to load analytics"}
        </p>
        <button
          onClick={load}
          className="mt-3 px-4 py-2 text-[13px] font-medium text-gold border border-gold/30 rounded-lg hover:bg-gold/10 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  const maxDaily = Math.max(...data.dailyCounts.map((d) => d.count), 1);

  // Build merged content type data: count + engagement rate in one view
  const mergedContentTypes = data.byContentType.map((ct) => {
    const perfMatch = perf?.byContentType.find((p) => p.contentType === ct.content_type);
    const total = data.byContentType.reduce((sum, c) => sum + c.count, 0);
    return {
      name: CONTENT_TYPE_LABELS[ct.content_type] ?? ct.content_type,
      count: ct.count,
      pct: total > 0 ? Math.round((ct.count / total) * 100) : 0,
      pubPct: ct.count > 0 ? Math.round((ct.published / ct.count) * 100) : 0,
      engRate: perfMatch?.avgEngagementRate ?? null,
    };
  });

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-end justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-semibold text-ink">Social Performance</h3>
          <p className="text-[14px] text-ink-faint mt-0.5">
            {account === "all" ? "All accounts" : `@${account}`} &middot; Posting and engagement metrics
          </p>
        </div>
      </div>

      {/* KPI cards — single merged row */}
      <div className={`grid grid-cols-2 gap-3 sm:grid-cols-4 transition-opacity ${perfLoading ? "opacity-50" : ""}`}>
        {[
          {
            label: "Engagement",
            value: perf ? `${perf.overall.avgEngagementRate.toFixed(1)}%` : perfError ? "--" : "--",
            sub: perf ? `${TREND_ICON[perf.overall.recentTrend]} ${perf.overall.recentTrend}` : perfError ? "failed to load" : "loading",
            color: "text-ink",
          },
          {
            label: "Published",
            value: data.totalPublished,
            sub: `${data.publishedThisWeek} this week`,
            color: "text-type-lesson",
          },
          {
            label: "Impressions",
            value: perf
              ? perf.topPerformers.reduce((s, t) => s + t.impressions, 0).toLocaleString()
              : "--",
            sub: perf ? `${perf.overall.totalTracked} tracked` : "loading",
            color: "text-ink",
          },
          {
            label: "Replies",
            value: data.engagement.replies_sent,
            sub: `${data.engagement.replies_pending} pending`,
            color: "text-[#0A66C2]",
          },
        ].map((kpi) => (
          <div key={kpi.label} className="bg-surface border border-edge rounded-xl p-3.5">
            <div className={`text-[22px] font-semibold ${kpi.color}`}>{kpi.value}</div>
            <div className="text-[14px] font-medium text-ink mt-0.5">{kpi.label}</div>
            <div className="text-[12px] text-ink-faint mt-0.5">{kpi.sub}</div>
          </div>
        ))}
      </div>

      {/* Daily activity chart (last 28 days) */}
      {data.dailyCounts.length > 0 && (
        <div className="bg-surface border border-edge rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[14px] font-medium text-ink">Daily Posts (28d)</span>
            <span className="text-[12px] text-ink-faint">
              {data.dailyCounts.reduce((sum, d) => sum + d.count, 0)} total
            </span>
          </div>
          <div className="flex items-end gap-[3px] h-16">
            {data.dailyCounts.map((d) => {
              const height = d.count > 0 ? Math.max((d.count / maxDaily) * 100, 8) : 0;
              const isToday = d.date === new Date().toISOString().slice(0, 10);
              return (
                <div
                  key={d.date}
                  className="flex-1 group relative"
                  title={`${d.date}: ${d.count} posts`}
                >
                  <div
                    className={`w-full rounded-sm transition-colors ${
                      d.count === 0
                        ? "bg-ink-faint/10"
                        : isToday
                          ? "bg-type-decision"
                          : "bg-type-decision/40 group-hover:bg-type-decision/60"
                    }`}
                    style={{ height: d.count === 0 ? "2px" : `${height}%` }}
                  />
                </div>
              );
            })}
          </div>
          <div className="flex justify-between mt-2 text-[11px] text-ink-faint">
            <span>{data.dailyCounts[0]?.date.slice(5)}</span>
            <span>Today</span>
          </div>
        </div>
      )}

      {/* Content Performance — merged count + engagement */}
      {mergedContentTypes.length > 0 && (
        <div className="bg-surface border border-edge rounded-xl p-4">
          <span className="text-[14px] font-medium text-ink block mb-3">Content Performance</span>
          <div className="space-y-2.5">
            {mergedContentTypes.map((ct) => (
              <div key={ct.name} className="flex items-center gap-3">
                <span className="text-[14px] text-ink w-28 truncate">{ct.name}</span>
                <div className="flex-1 h-5 bg-ink-faint/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-type-decision/40 rounded-full transition-all"
                    style={{ width: `${ct.pct}%` }}
                  />
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[12px] text-ink-faint w-12 text-right">
                    {ct.count} ({ct.pubPct}%)
                  </span>
                  {ct.engRate !== null && (
                    <span className="text-[12px] text-type-decision font-medium w-14 text-right">
                      {ct.engRate.toFixed(1)}% eng
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top Tweets — merged performers + recent, sorted by engagement */}
      {perf && perf.topPerformers.length > 0 && (
        <div className={`bg-surface border border-edge rounded-xl p-4 transition-opacity ${perfLoading ? "opacity-50" : ""}`}>
          <span className="text-[14px] font-medium text-ink block mb-3">Top Tweets (30d)</span>
          <div className="space-y-1">
            {perf.topPerformers.map((t, i) => (
              <div key={i} className="flex items-start gap-3 py-2.5 border-b border-edge last:border-0">
                <div className="min-w-0 flex-1">
                  <p className="text-[14px] text-ink leading-relaxed line-clamp-2">{t.text}</p>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1.5 text-[12px] text-ink-faint">
                    <span className="text-type-decision font-medium">{t.engagementRate.toFixed(1)}% engagement</span>
                    <span>{t.impressions.toLocaleString()} views</span>
                    <span>{t.likes} likes</span>
                    <span>{t.retweets} reposts</span>
                    <span>{t.replies} replies</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

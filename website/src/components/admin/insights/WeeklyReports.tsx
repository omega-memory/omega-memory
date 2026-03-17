import { useCallback, useEffect, useRef, useState } from "react";
import CollapsibleSection from "./CollapsibleSection";

interface Report {
  id: string;
  account: string;
  period_start: string;
  period_end: string;
  metrics: {
    overall: { totalTracked: number; avgEngagementRate: number; recentTrend: string };
    topPerformers: { text: string; engagementRate: number; impressions: number }[];
  };
  narrative: string | null;
  recommendations: string[] | null;
  emailed_at: string | null;
  created_at: string;
}

interface WeeklyReportsProps {
  account: "all" | "jasonsosa" | "omega_memory";
}

export default function WeeklyReports({ account: rawAccount }: WeeklyReportsProps) {
  const account = rawAccount === "all" ? "omega_memory" : rawAccount;
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchReports = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/admin/reports?account=${account}&limit=10`);
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      setReports(data.reports || []);
    } catch {
      setError("Failed to load reports");
    }
    setLoading(false);
  }, [account]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStartRef = useRef<number>(0);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setGenerating(false);
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    const prevTopId = reports[0]?.id ?? null;

    try {
      const res = await fetch("/api/admin/reports", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      if (!res.ok) throw new Error(`${res.status}`);

      // Poll fetchReports every 5s for up to 60s until a new report appears
      pollStartRef.current = Date.now();
      pollRef.current = setInterval(async () => {
        const elapsed = Date.now() - pollStartRef.current;
        if (elapsed >= 60000) {
          stopPolling();
          setError("Report generation timed out. It may still be processing.");
          return;
        }
        try {
          const pollRes = await fetch(`/api/admin/reports?account=${account}&limit=10`);
          if (!pollRes.ok) return;
          const data = await pollRes.json();
          const newReports = data.reports || [];
          // Compare most-recent report ID rather than count — count-based
          // detection fails when the list is already at the API limit cap.
          if (newReports.length > 0 && newReports[0]?.id !== prevTopId) {
            setReports(newReports);
            stopPolling();
          }
        } catch {
          // Keep polling on transient errors
        }
      }, 5000);
    } catch {
      setError("Failed to schedule report");
      setGenerating(false);
    }
  };

  const trendIcon = (trend: string) => {
    if (trend === "improving") return <span className="text-type-success">&#x25B2;</span>;
    if (trend === "declining") return <span className="text-type-error">&#x25BC;</span>;
    return <span className="text-ink-tertiary">&#x2013;</span>;
  };

  return (
    <CollapsibleSection label="Weekly Reports" summary={`@${account}`}>
      <div className="space-y-3">
        {/* Generate button */}
        <div className="flex items-center justify-between">
          <p className="text-[12px] text-ink-tertiary">
            {reports.length > 0 ? `${reports.length} reports` : "No reports yet"}
          </p>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="px-3 py-2 bg-gold text-surface-base text-[13px] font-semibold rounded-lg hover:bg-gold/90 transition-colors disabled:opacity-50 min-h-[36px]"
          >
            {generating ? (
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generating report...
              </span>
            ) : "Generate Now"}
          </button>
        </div>

        {error && (
          <div className="p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
            {error}
          </div>
        )}

        {loading ? (
          <div className="space-y-2">
            {[0, 1].map((i) => (
              <div key={i} className="p-3 admin-card">
                <div className="skeleton h-4 w-32 rounded mb-2" />
                <div className="skeleton h-3 w-20 rounded" />
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {reports.map((report) => (
              <div key={report.id} className="admin-card overflow-hidden">
                <button
                  onClick={() => setExpanded(expanded === report.id ? null : report.id)}
                  className="w-full p-3 flex items-center justify-between text-left hover:bg-surface-hover transition-colors"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[14px] font-medium text-ink">
                        {new Date(report.period_start).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} to {new Date(report.period_end).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                      </span>
                      {report.emailed_at && (
                        <span className="px-1.5 py-0.5 bg-type-success/10 text-type-success text-[10px] rounded">
                          emailed
                        </span>
                      )}
                    </div>
                    {report.metrics?.overall && (
                    <div className="flex items-center gap-3 mt-1 text-[12px] text-ink-tertiary font-mono">
                      <span>{report.metrics.overall.totalTracked} tweets</span>
                      <span>{report.metrics.overall.avgEngagementRate}% eng</span>
                      <span>{trendIcon(report.metrics.overall.recentTrend)}</span>
                    </div>
                    )}
                  </div>
                  <svg
                    className={`w-4 h-4 text-ink-tertiary transition-transform ${expanded === report.id ? "rotate-180" : ""}`}
                    fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                  </svg>
                </button>

                {expanded === report.id && (
                  <div className="px-3 pb-3 space-y-3 border-t border-edge">
                    {/* Narrative */}
                    {report.narrative && (
                      <div className="pt-3">
                        <h4 className="text-[12px] font-semibold text-ink-secondary mb-2">Analysis</h4>
                        {report.narrative.split("\n\n").map((para, i) => (
                          <p key={i} className="text-[13px] text-ink-secondary leading-relaxed mb-2">{para}</p>
                        ))}
                      </div>
                    )}

                    {/* Recommendations */}
                    {report.recommendations && report.recommendations.length > 0 && (
                      <div>
                        <h4 className="text-[12px] font-semibold text-ink-secondary mb-2">Recommendations</h4>
                        <ul className="space-y-1">
                          {report.recommendations.map((rec, i) => (
                            <li key={i} className="text-[13px] text-ink-tertiary flex gap-2">
                              <span className="text-gold font-semibold shrink-0">{i + 1}.</span>
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Top Tweets */}
                    {report.metrics.topPerformers.length > 0 && (
                      <div>
                        <h4 className="text-[12px] font-semibold text-ink-secondary mb-2">Top Tweets</h4>
                        {report.metrics.topPerformers.slice(0, 3).map((tweet, i) => (
                          <div key={i} className="p-2 bg-surface rounded-lg mb-1.5">
                            <p className="text-[12px] text-ink-secondary line-clamp-2">{tweet.text}</p>
                            <p className="text-[11px] text-ink-tertiary font-mono mt-1">
                              {tweet.engagementRate}% eng, {tweet.impressions.toLocaleString()} imp
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}

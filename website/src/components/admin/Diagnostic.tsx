import { useState, useEffect, useCallback } from "react";
import type { DiagnosticData } from "./lib/types";
import { useProjects } from "./hooks/useProjects";

const PERIODS = [7, 30, 90] as const;
type Period = (typeof PERIODS)[number];

function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function formatCost(n: number): string {
  return "$" + n.toFixed(2);
}

function VerdictBadge({ verdict }: { verdict: "healthy" | "underused" | "idle" }) {
  const styles = {
    healthy: "bg-type-lesson/10 text-type-lesson border-type-lesson/20",
    underused: "bg-gold/10 text-gold border-gold/20",
    idle: "bg-type-error/10 text-type-error border-type-error/20",
  };
  const labels = { healthy: "Healthy", underused: "Underused", idle: "Idle" };
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[12px] font-semibold border ${styles[verdict]}`}>
      <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
        verdict === "healthy" ? "bg-type-lesson" : verdict === "underused" ? "bg-gold" : "bg-type-error"
      }`} />
      {labels[verdict]}
    </span>
  );
}

function SummaryCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="p-4 admin-card">
      <div className="text-[11px] uppercase tracking-wider text-ink-faint mb-1">{label}</div>
      <div className="text-2xl font-semibold text-ink tabular-nums">{value}</div>
      {sub && <div className="text-[12px] text-ink-tertiary mt-0.5">{sub}</div>}
    </div>
  );
}

function HBar({ label, value, max, highlight }: { label: string; value: number; max: number; highlight?: boolean }) {
  const pct = max > 0 ? Math.max((value / max) * 100, 0.5) : 0;
  return (
    <div className="flex items-center gap-3 py-1">
      <div className={`w-40 text-[12px] truncate ${highlight ? "text-gold font-medium" : "text-ink-secondary"}`}>
        {label}
      </div>
      <div className="flex-1 h-5 bg-surface-elevated rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all duration-300 ${highlight ? "bg-gold/40" : "bg-ink-faint/30"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className={`w-16 text-right text-[12px] tabular-nums ${highlight ? "text-gold" : "text-ink-tertiary"}`}>
        {value.toLocaleString()}
      </div>
    </div>
  );
}

function StackedBar({ buckets }: { buckets: { never: number; low: number; medium: number; high: number } }) {
  const total = buckets.never + buckets.low + buckets.medium + buckets.high;
  if (total === 0) return null;
  const segments = [
    { key: "never", count: buckets.never, color: "bg-type-error/40", label: "Never" },
    { key: "low", count: buckets.low, color: "bg-gold/30", label: "Low (1-2)" },
    { key: "medium", count: buckets.medium, color: "bg-gold/50", label: "Med (3-9)" },
    { key: "high", count: buckets.high, color: "bg-type-lesson/40", label: "High (10+)" },
  ];
  return (
    <div>
      <div className="h-6 flex rounded overflow-hidden">
        {segments.map((s) => {
          const pct = (s.count / total) * 100;
          if (pct < 0.5) return null;
          return (
            <div
              key={s.key}
              className={`${s.color} transition-all duration-300`}
              style={{ width: `${pct}%` }}
              title={`${s.label}: ${s.count} (${pct.toFixed(0)}%)`}
            />
          );
        })}
      </div>
      <div className="flex gap-4 mt-2">
        {segments.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5 text-[11px] text-ink-tertiary">
            <span className={`w-2 h-2 rounded-sm ${s.color}`} />
            {s.label}: {s.count}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Diagnostic() {
  const { resolvePathToProject } = useProjects();
  const [data, setData] = useState<DiagnosticData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [period, setPeriod] = useState<Period>(30);

  const fetchDiagnostic = useCallback(async (days: number) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`/api/admin/diagnostic?days=${days}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDiagnostic(period);
  }, [period, fetchDiagnostic]);

  if (loading && !data) {
    return (
      <div className="px-5 pt-6 space-y-4">
        <div className="h-8 w-48 rounded-lg skeleton" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 rounded-xl skeleton" />
          ))}
        </div>
        <div className="h-32 rounded-xl skeleton" />
        <div className="h-48 rounded-xl skeleton" />
        <div className="h-40 rounded-xl skeleton" />
        <div className="h-48 rounded-xl skeleton" />
        <div className="h-40 rounded-xl skeleton" />
        <div className="h-48 rounded-xl skeleton" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="px-5 pt-6">
        <div className="p-4 bg-type-error/10 border border-type-error/20 rounded-xl text-[13px] text-type-error">
          Failed to load diagnostic data: {error}
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { memory_health: mem, tool_usage: tools, sessions, llm_costs: costs, quality, latency } = data;
  const topToolMax = tools.top_tools.length > 0 ? tools.top_tools[0].calls : 1;

  return (
    <div className="px-5 pt-6 pb-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h2 className="admin-heading !mb-0">System Diagnostic</h2>
          <VerdictBadge verdict={data.verdict} />
        </div>
        <div className="flex items-center gap-2">
          {/* Period toggle */}
          <div className="flex rounded-lg border border-edge overflow-hidden">
            {PERIODS.map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1.5 text-[12px] font-medium transition-colors ${
                  period === p
                    ? "bg-gold/10 text-gold"
                    : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
                }`}
              >
                {p}d
              </button>
            ))}
          </div>
          {/* Refresh */}
          <button
            onClick={() => fetchDiagnostic(period)}
            disabled={loading}
            className="p-2 rounded-lg text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover transition-colors disabled:opacity-50"
            title="Refresh"
          >
            <svg className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
            </svg>
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <SummaryCard
          label="Memory Utilization"
          value={`${mem.utilization_pct}%`}
          sub={`${mem.total.toLocaleString()} total memories`}
        />
        <SummaryCard
          label="7-Day Velocity"
          value={mem.velocity_total_7d.toLocaleString()}
          sub={mem.velocity_7d.length > 0
            ? `Top: ${mem.velocity_7d[0].event_type} (${mem.velocity_7d[0].count})`
            : "No activity"
          }
        />
        <SummaryCard
          label="OMEGA Tool Calls"
          value={tools.omega_calls.toLocaleString()}
          sub={`${tools.omega_per_session}/session avg`}
        />
        <SummaryCard
          label={`Sessions (${period}d)`}
          value={sessions.period}
          sub={`${sessions.week} this week`}
        />
      </div>

      {/* Recommendations */}
      <section>
        <div className="admin-section-label">Recommendations</div>
        <div className="admin-card p-4 space-y-2">
          {data.recommendations.map((rec, i) => (
            <div key={i} className={`flex items-start gap-3 p-3 rounded-lg ${
              rec.severity === "critical" ? "bg-type-error/10 border border-type-error/20" :
              rec.severity === "warning" ? "bg-gold/10 border border-gold/20" :
              "bg-surface-elevated border border-edge"
            }`}>
              <span className={`mt-0.5 w-2 h-2 rounded-full shrink-0 ${
                rec.severity === "critical" ? "bg-type-error" :
                rec.severity === "warning" ? "bg-gold" :
                "bg-ink-faint"
              }`} />
              <div>
                <div className={`text-[13px] ${
                  rec.severity === "critical" ? "text-type-error" :
                  rec.severity === "warning" ? "text-gold" :
                  "text-ink-secondary"
                }`}>{rec.message}</div>
                {rec.action && (
                  <div className="text-[11px] text-ink-faint mt-0.5">{rec.action}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Memory Health */}
      <section>
        <div className="admin-section-label">Memory Health <span className="text-ink-faint font-normal text-[10px]">all-time</span></div>
        <div className="admin-card p-4 space-y-4">
          <div>
            <div className="text-[12px] text-ink-tertiary mb-2">Access Distribution</div>
            <StackedBar buckets={mem.access_buckets} />
          </div>
          <div className="admin-divider" />
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[12px] text-ink-tertiary">Dead Memories</div>
              <div className="text-[11px] text-ink-faint">Never accessed, older than 14 days</div>
            </div>
            <div className={`text-lg font-semibold tabular-nums ${mem.dead_pct > 20 ? "text-type-error" : "text-ink"}`}>
              {mem.dead_memories.toLocaleString()}
              <span className="text-[12px] font-normal text-ink-tertiary ml-1">({mem.dead_pct}%)</span>
            </div>
          </div>
          {mem.dead_by_type.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {mem.dead_by_type.map((d) => (
                <span key={d.event_type} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-type-error/10 text-[11px] text-type-error/80">
                  <span>{d.event_type}</span>
                  <span className="font-medium tabular-nums">{d.count}</span>
                </span>
              ))}
            </div>
          )}
          {mem.velocity_7d.length > 0 && (
            <>
              <div className="admin-divider" />
              <div>
                <div className="text-[12px] text-ink-tertiary mb-2">7-Day Velocity by Type</div>
                <div className="flex flex-wrap gap-2">
                  {mem.velocity_7d.slice(0, 8).map((v) => (
                    <span key={v.event_type} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-surface-elevated text-[11px] text-ink-secondary">
                      <span className="text-ink-faint">{v.event_type}</span>
                      <span className="font-medium tabular-nums">{v.count}</span>
                    </span>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Memory Quality */}
      <section>
        <div className="admin-section-label">Memory Quality <span className="text-ink-faint font-normal text-[10px]">all-time</span></div>
        <div className="admin-card p-4 space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Avg Priority</div>
              <div className={`text-xl font-semibold tabular-nums ${data.quality.avg_priority < 2.5 ? "text-gold" : "text-ink"}`}>{data.quality.avg_priority.toFixed(1)}<span className="text-[12px] text-ink-tertiary">/5</span></div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">With Tags</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{data.quality.tags_pct}%</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">With Project</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{data.quality.project_pct}%</div>
            </div>
          </div>
          {data.quality.priority_distribution.length > 0 && (
            <>
              <div className="admin-divider" />
              <div>
                <div className="text-[12px] text-ink-tertiary mb-2">Priority Distribution</div>
                <div className="flex gap-2">
                  {data.quality.priority_distribution.map((p) => (
                    <div key={p.priority} className="flex-1 text-center">
                      <div className="text-[18px] font-semibold text-ink tabular-nums">{p.count}</div>
                      <div className="text-[11px] text-ink-faint">P{p.priority}</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
          {data.quality.by_project.length > 0 && (
            <>
              <div className="admin-divider" />
              <div>
                <div className="text-[12px] text-ink-tertiary mb-2">By Project</div>
                <div className="space-y-0.5">
                  {data.quality.by_project.slice(0, 8).map((p) => {
                    const resolved = resolvePathToProject(p.project);
                    const label = resolved?.name ?? (p.project || "(unscoped)");
                    return <HBar key={p.project} label={label} value={p.count} max={data.quality.by_project[0].count} />;
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Tool Usage */}
      <section>
        <div className="admin-section-label">Tool Usage ({period}d)</div>
        <div className="admin-card p-4 space-y-4">
          <div>
            <div className="text-[12px] text-ink-tertiary mb-2">Top Tools</div>
            <div className="space-y-0.5">
              {tools.top_tools.slice(0, 10).map((t) => (
                <HBar
                  key={t.tool}
                  label={t.tool}
                  value={t.calls}
                  max={topToolMax}
                  highlight={t.tool.startsWith("mcp__omega")}
                />
              ))}
            </div>
          </div>
          {tools.omega_tools.length > 0 && (
            <>
              <div className="admin-divider" />
              <div>
                <div className="text-[12px] text-ink-tertiary mb-2">OMEGA Tools</div>
                <div className="space-y-0.5">
                  {tools.omega_tools.map((t) => (
                    <HBar
                      key={t.tool}
                      label={t.tool.replace("mcp__omega__", "")}
                      value={t.calls}
                      max={tools.omega_tools[0].calls}
                      highlight
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Latency */}
      <section>
        <div className="admin-section-label">Latency ({period}d)</div>
        <div className="admin-card p-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Average</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{Math.round(data.latency.avg_ms)}ms</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">P95</div>
              <div className={`text-xl font-semibold tabular-nums ${data.latency.p95_ms > 1000 ? "text-type-error" : "text-ink"}`}>{Math.round(data.latency.p95_ms)}ms</div>
            </div>
          </div>
          {data.latency.by_tool.length > 0 && (
            <>
              <div className="admin-divider" />
              <div>
                <div className="text-[12px] text-ink-tertiary mb-2">By Tool (top 10)</div>
                <div className="space-y-1">
                  {data.latency.by_tool.slice(0, 10).map((t) => (
                    <div key={t.tool} className="flex items-center justify-between py-1 text-[12px]">
                      <span className="text-ink-secondary truncate max-w-[50%]">{t.tool}</span>
                      <div className="flex items-center gap-4">
                        <span className="text-ink-tertiary tabular-nums">{t.calls} calls</span>
                        <span className="text-ink-faint tabular-nums w-16 text-right">avg {Math.round(t.avg_ms)}ms</span>
                        <span className={`tabular-nums w-16 text-right font-medium ${t.p95_ms > 1000 ? "text-type-error" : "text-ink"}`}>p95 {Math.round(t.p95_ms)}ms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* LLM Costs */}
      <section>
        <div className="admin-section-label">LLM Costs ({period}d)</div>
        <div className="admin-card p-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Total Spend</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{formatCost(costs.total_usd)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Input Tokens</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{formatTokens(costs.total_input_tokens)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Output Tokens</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{formatTokens(costs.total_output_tokens)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-faint">Sessions</div>
              <div className="text-xl font-semibold text-ink tabular-nums">{costs.total_sessions}</div>
            </div>
          </div>
          {costs.by_model.length > 0 && (
            <>
              <div className="admin-divider" />
              <div className="mt-4">
                <div className="text-[12px] text-ink-tertiary mb-2">Cost by Model</div>
                <div className="space-y-1.5">
                  {costs.by_model.map((m) => (
                    <div key={m.model} className="flex items-center justify-between py-1 text-[12px]">
                      <span className="text-ink-secondary font-mono truncate max-w-[60%]">{m.model}</span>
                      <div className="flex items-center gap-4">
                        <span className="text-ink-tertiary tabular-nums">{m.calls} sessions</span>
                        <span className="text-ink font-medium tabular-nums w-16 text-right">{formatCost(m.cost)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

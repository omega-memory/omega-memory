import { useCallback, useEffect, useState } from "react";

import { formatCost, formatTokens } from "./jobUtils";

interface CostSummary {
  totalRuns: number;
  okCount: number;
  errorCount: number;
  totalCostUsd: number;
  avgCostUsd: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  avgDurationMs: number;
  periodDays: number;
}

interface CostSeriesPoint {
  date: string;
  cost: number;
  status: string;
}

interface CostTabProps {
  scheduleLabel: string;
}

export function CostTab({ scheduleLabel }: CostTabProps) {
  const [summary, setSummary] = useState<CostSummary | null>(null);
  const [series, setSeries] = useState<CostSeriesPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCosts = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/admin/schedule-runs?label=${encodeURIComponent(scheduleLabel)}&mode=summary&days=30`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setSummary(data?.summary ?? null);
        setSeries(data?.costSeries ?? []);
      })
      .catch(() => setError("Failed to load cost data"))
      .finally(() => setLoading(false));
  }, [scheduleLabel]);

  useEffect(() => {
    fetchCosts();
  }, [fetchCosts]);

  if (loading) {
    return (
      <div className="space-y-4 pt-4">
        <div className="h-20 rounded-lg skeleton" />
        <div className="h-32 rounded-lg skeleton" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-10">
        <p className="text-[15px] text-type-error">{error}</p>
        <button
          onClick={fetchCosts}
          className="mt-3 text-[14px] text-ink-tertiary hover:text-ink-secondary underline transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!summary || summary.totalRuns === 0) {
    return (
      <div className="text-center py-10 text-[15px] text-ink-tertiary">
        No cost data yet. Costs will appear after runs are recorded.
      </div>
    );
  }

  const maxCost = Math.max(...series.map((s) => s.cost), 0.001);

  return (
    <div className="space-y-5 pt-2">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface-elevated rounded-lg p-3 border border-edge/50">
          <div className="text-[12px] text-ink-faint uppercase tracking-wider">Total (30d)</div>
          <div className="text-[22px] font-semibold text-ink mt-1">{formatCost(summary.totalCostUsd)}</div>
          <div className="text-[13px] text-ink-tertiary">{summary.totalRuns} runs</div>
        </div>
        <div className="bg-surface-elevated rounded-lg p-3 border border-edge/50">
          <div className="text-[12px] text-ink-faint uppercase tracking-wider">Avg / Run</div>
          <div className="text-[22px] font-semibold text-ink mt-1">{formatCost(summary.avgCostUsd)}</div>
          <div className="text-[13px] text-ink-tertiary">
            {summary.okCount} ok, {summary.errorCount} err
          </div>
        </div>
      </div>

      {/* Token breakdown */}
      <div className="bg-surface-elevated rounded-lg p-3 border border-edge/50">
        <div className="text-[12px] text-ink-faint uppercase tracking-wider mb-2">Tokens (30d)</div>
        <div className="flex gap-6">
          <div>
            <div className="text-[18px] font-semibold text-ink">{formatTokens(summary.totalInputTokens)}</div>
            <div className="text-[12px] text-ink-tertiary">input</div>
          </div>
          <div>
            <div className="text-[18px] font-semibold text-ink">{formatTokens(summary.totalOutputTokens)}</div>
            <div className="text-[12px] text-ink-tertiary">output</div>
          </div>
        </div>
      </div>

      {/* Mini bar chart */}
      {series.length > 1 && (
        <div>
          <div className="text-[12px] text-ink-faint uppercase tracking-wider mb-2">Cost per run</div>
          <div className="flex items-end gap-[2px] h-16">
            {series.slice(0, 30).map((point, i) => {
              const h = Math.max(2, (point.cost / maxCost) * 60);
              return (
                <div
                  key={point.date + i}
                  title={`${new Date(point.date).toLocaleDateString()} — ${formatCost(point.cost)}`}
                  className={`flex-1 rounded-t-sm transition-colors ${
                    point.status === "error" ? "bg-type-error/60" : "bg-type-decision/40"
                  }`}
                  style={{ height: `${h}px` }}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

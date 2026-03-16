import { useState, useEffect, useCallback } from "react";

interface AlertRecord {
  id: string;
  type: string;
  severity: string;
  title: string;
  status: string;
  created_at: string;
  resolved_at: string | null;
}

interface AlertHistoryData {
  alerts: AlertRecord[];
  recurrenceCounts: Record<string, number>;
}

const SEVERITY_STYLES: Record<string, string> = {
  critical: "bg-type-error/10 text-type-error border-type-error/20",
  warning: "bg-gold/10 text-gold border-gold/20",
  info: "bg-surface-elevated text-ink-secondary border-edge",
};

const STATUS_STYLES: Record<string, { dot: string; label: string }> = {
  active: { dot: "bg-gold", label: "Active" },
  resolved: { dot: "bg-type-lesson", label: "Resolved" },
  dismissed: { dot: "bg-ink-faint", label: "Dismissed" },
  snoozed: { dot: "bg-[#60a5fa]", label: "Snoozed" },
};

function timeAgo(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function AlertHistory() {
  const [data, setData] = useState<AlertHistoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "resolved">("all");

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/admin/alert-history?status=${statusFilter}`);
      if (res.ok) setData(await res.json());
    } catch {
      // Non-critical
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => { fetchAlerts(); }, [fetchAlerts]);

  if (loading && !data) {
    return (
      <section>
        <div className="admin-section-label">Alert History</div>
        <div className="admin-card h-32 skeleton rounded-xl" />
      </section>
    );
  }

  if (!data || data.alerts.length === 0) {
    return (
      <section>
        <div className="admin-section-label">Alert History</div>
        <div className="admin-card px-4 py-8 text-center text-[13px] text-ink-faint">
          No alerts recorded yet
        </div>
      </section>
    );
  }

  // Check for recurring patterns
  const patterns = Object.entries(data.recurrenceCounts)
    .filter(([, count]) => count >= 3)
    .sort((a, b) => b[1] - a[1]);

  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <div className="admin-section-label !mb-0">Alert History</div>
        <div className="flex rounded-lg border border-edge overflow-hidden">
          {(["all", "active", "resolved"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-2.5 py-1 text-[11px] font-medium transition-colors capitalize ${
                statusFilter === s ? "bg-gold/10 text-gold" : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Recurring pattern alerts */}
      {patterns.length > 0 && (
        <div className="mb-3 space-y-1.5">
          {patterns.map(([type, count]) => (
            <div key={type} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gold/[0.03] border border-gold/[0.06]">
              <span className="text-[11px] font-mono text-gold/60">Pattern</span>
              <span className="text-[13px] text-ink-secondary flex-1">
                {type.replace(/_/g, " ")} has occurred {count} times this week
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Alert list */}
      <div className="admin-card divide-y divide-edge">
        {data.alerts.map((alert) => {
          const sevStyle = SEVERITY_STYLES[alert.severity] ?? SEVERITY_STYLES.info;
          const statStyle = STATUS_STYLES[alert.status] ?? STATUS_STYLES.active;

          return (
            <div key={alert.id} className="flex items-center gap-3 px-4 py-3">
              <span className={`w-2 h-2 rounded-full shrink-0 ${statStyle.dot}`} />
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-medium border ${sevStyle}`}>
                {alert.severity}
              </span>
              <span className="text-[13px] text-ink-secondary flex-1 truncate">{alert.title}</span>
              <span className="text-[11px] text-ink-faint tabular-nums shrink-0">{timeAgo(alert.created_at)}</span>
              {alert.resolved_at && (
                <span className="text-[10px] text-type-lesson/60 shrink-0">resolved {timeAgo(alert.resolved_at)}</span>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

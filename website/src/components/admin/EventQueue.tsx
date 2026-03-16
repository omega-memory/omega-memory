import { useCallback, useEffect, useState } from "react";

interface PendingEvent {
  id: string;
  account: string | null;
  event_type: string;
  status: string;
  scheduled_for: string;
  recurrence: string | null;
  payload: Record<string, unknown>;
  attempts: number;
  max_attempts: number;
  last_error: string | null;
  created_at: string;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-yellow-500/20 text-yellow-400",
  processing: "bg-blue-500/20 text-blue-400",
  completed: "bg-type-success/20 text-type-success",
  failed: "bg-type-error/20 text-type-error",
  cancelled: "bg-ink-tertiary/20 text-ink-tertiary",
};

interface EventQueueProps {
  account?: "all" | "jasonsosa" | "omega_memory";
  onCountChange?: (count: number) => void;
}

export default function EventQueue({ account, onCountChange }: EventQueueProps) {
  const [events, setEvents] = useState<PendingEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [error, setError] = useState<string | null>(null);

  // Schedule form
  const [showForm, setShowForm] = useState(false);
  const [formType, setFormType] = useState("send_weekly_report");
  const [formSchedule, setFormSchedule] = useState("");
  const [formPayload, setFormPayload] = useState("{}");

  const fetchEvents = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (account && account !== "all") params.set("account", account);
      if (statusFilter !== "all") params.set("status", statusFilter);
      const res = await fetch(`/api/admin/events?${params}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      const items: PendingEvent[] = data.events || [];
      setEvents(items);
      const pendingCount = items.filter((e) => e.status === "pending" || e.status === "processing").length;
      onCountChange?.(pendingCount);
    } catch {
      setError("Failed to load events");
    }
    setLoading(false);
  }, [account, statusFilter, onCountChange]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const handleAction = async (id: string, action: "cancel" | "run_now") => {
    try {
      const res = await fetch("/api/admin/events", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, action }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      fetchEvents();
    } catch {
      setError(`Failed to ${action} event`);
    }
  };

  const handleCreate = async () => {
    if (!formSchedule) return;
    try {
      let payload = {};
      try { payload = JSON.parse(formPayload); } catch { /* ignore */ }

      const res = await fetch("/api/admin/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account: account && account !== "all" ? account : null,
          event_type: formType,
          scheduled_for: new Date(formSchedule).toISOString(),
          payload,
          source: "admin",
        }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      setShowForm(false);
      setFormPayload("{}");
      fetchEvents();
    } catch {
      setError("Failed to create event");
    }
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
      " " + d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  };

  return (
    <div className="px-5 pt-4 pb-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-[16px] font-semibold text-ink">Event Queue</h3>
        <div className="flex gap-2">
          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink focus:border-gold focus:outline-none min-h-[44px]"
          >
            <option value="all">All statuses</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <button
            onClick={() => setShowForm(!showForm)}
            className="px-4 py-2 bg-gold text-surface-base text-[13px] font-semibold rounded-lg hover:bg-gold/90 transition-colors min-h-[44px]"
          >
            Schedule Event
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
          {error}
          <button onClick={() => setError(null)} className="ml-2 underline">dismiss</button>
        </div>
      )}

      {/* Create form */}
      {showForm && (
        <div className="mb-4 p-4 admin-card space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[12px] font-medium text-ink-secondary mb-1">Event Type</label>
              <select
                value={formType}
                onChange={(e) => setFormType(e.target.value)}
                className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink focus:border-gold focus:outline-none"
              >
                <option value="send_weekly_report">Send Weekly Report</option>
                <option value="send_tweet">Send Tweet</option>
                <option value="check_engagement">Check Engagement</option>
              </select>
            </div>
            <div>
              <label className="block text-[12px] font-medium text-ink-secondary mb-1">Scheduled For</label>
              <input
                type="datetime-local"
                value={formSchedule}
                onChange={(e) => setFormSchedule(e.target.value)}
                className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink focus:border-gold focus:outline-none"
              />
            </div>
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-secondary mb-1">Payload (JSON)</label>
            <input
              value={formPayload}
              onChange={(e) => setFormPayload(e.target.value)}
              placeholder='{"tweet_id": "..."}'
              className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink font-mono focus:border-gold focus:outline-none"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              className="px-4 py-2 bg-gold text-surface-base text-[13px] font-semibold rounded-lg hover:bg-gold/90 transition-colors"
            >
              Create
            </button>
            <button
              onClick={() => setShowForm(false)}
              className="px-4 py-2 text-ink-tertiary text-[13px] hover:text-ink transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Events list */}
      {loading ? (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="p-4 admin-card">
              <div className="skeleton h-4 w-40 rounded mb-2" />
              <div className="skeleton h-3 w-24 rounded" />
            </div>
          ))}
        </div>
      ) : events.length === 0 ? (
        <div className="p-8 text-center text-[14px] text-ink-tertiary">
          No events found
        </div>
      ) : (
        <div className="space-y-2">
          {events.map((event) => (
            <div key={event.id} className="p-4 admin-card">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${STATUS_COLORS[event.status] || ""}`}>
                    {event.status}
                  </span>
                  <span className="text-[14px] font-medium text-ink">{event.event_type}</span>
                  {event.recurrence && (
                    <span className="px-2 py-0.5 bg-surface-elevated rounded text-[11px] text-ink-tertiary">
                      {event.recurrence}
                    </span>
                  )}
                </div>
                <div className="flex gap-1.5">
                  {event.status === "pending" && (
                    <>
                      <button
                        onClick={() => handleAction(event.id, "run_now")}
                        className="px-2.5 py-1.5 text-[12px] text-gold hover:bg-gold/10 rounded-md transition-colors"
                      >
                        Run now
                      </button>
                      <button
                        onClick={() => handleAction(event.id, "cancel")}
                        className="px-2.5 py-1.5 text-[12px] text-type-error hover:bg-type-error/10 rounded-md transition-colors"
                      >
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 text-[12px] text-ink-tertiary font-mono">
                <span>Scheduled: {formatDate(event.scheduled_for)}</span>
                {event.account && <span>@{event.account}</span>}
                {event.attempts > 0 && <span>Attempts: {event.attempts}/{event.max_attempts}</span>}
              </div>
              {event.last_error && (
                <div className="mt-2 p-2 bg-type-error/5 border border-type-error/10 rounded text-[12px] text-type-error/80 font-mono truncate">
                  {event.last_error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

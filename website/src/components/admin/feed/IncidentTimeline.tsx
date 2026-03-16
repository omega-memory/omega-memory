import { useState, useEffect, useCallback } from "react";
import type { TimelineData, TimelineEvent, TimelineEventSource, Tab } from "../lib/types";

const SOURCE_STYLES: Record<TimelineEventSource, { color: string; bg: string; label: string }> = {
  coordination: { color: "text-[#60a5fa]", bg: "bg-[#60a5fa]/10", label: "Coordination" },
  memory: { color: "text-gold", bg: "bg-gold/10", label: "Memory" },
  job: { color: "text-type-lesson", bg: "bg-type-lesson/10", label: "Job" },
  hook: { color: "text-[#c084fc]", bg: "bg-[#c084fc]/10", label: "Hook" },
  git: { color: "text-[#fb923c]", bg: "bg-[#fb923c]/10", label: "Git" },
};

function timeAgo(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function EventRow({ event, onNavigate }: { event: TimelineEvent; onNavigate?: (tab: Tab, id?: string) => void }) {
  const style = SOURCE_STYLES[event.source];
  const isError = event.eventType.includes("fail") || event.eventType.includes("error");
  const isCompleted = event.eventType.includes("complete") || event.eventType.includes("ok");

  return (
    <div className="flex items-start gap-3 group">
      {/* Timeline line + dot */}
      <div className="flex flex-col items-center shrink-0 w-6">
        <span className={`w-2.5 h-2.5 rounded-full mt-1 shrink-0 ${isError ? "bg-type-error" : isCompleted ? "bg-type-lesson" : "bg-ink-faint/30"}`} />
        <div className="w-px flex-1 bg-edge-subtle min-h-[24px]" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 pb-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-medium border border-transparent ${style.bg} ${style.color}`}>
            {style.label}
          </span>
          <span className="text-[13px] text-ink font-medium">{event.title}</span>
          <span className="text-[11px] text-ink-faint tabular-nums ml-auto shrink-0">{timeAgo(event.timestamp)}</span>
        </div>
        {event.detail && (
          <p className="text-[12px] text-ink-secondary mt-1 line-clamp-2">{event.detail}</p>
        )}
        <div className="flex items-center gap-3 mt-1">
          {event.agentId && (
            <span className="text-[11px] font-mono text-ink-faint">agent:{event.agentId.slice(0, 8)}</span>
          )}
          {event.project && (
            <span className="text-[11px] text-ink-faint">{event.project}</span>
          )}
          {event.linkedTab && onNavigate && (
            <button
              onClick={() => onNavigate(event.linkedTab!, event.linkedId)}
              className="text-[11px] text-gold/60 hover:text-gold transition-colors opacity-0 group-hover:opacity-100"
            >
              View &rarr;
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

interface IncidentTimelineProps {
  onNavigate?: (tab: Tab, id?: string) => void;
}

export default function IncidentTimeline({ onNavigate }: IncidentTimelineProps) {
  const [data, setData] = useState<TimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hours, setHours] = useState(24);
  const [sourceFilter, setSourceFilter] = useState<TimelineEventSource | "all">("all");

  const fetchTimeline = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        hours: String(hours),
        source: sourceFilter,
        limit: "50",
      });
      const res = await fetch(`/api/admin/timeline?${params}`);
      if (res.ok) setData(await res.json());
    } catch {
      // Non-critical
    } finally {
      setLoading(false);
    }
  }, [hours, sourceFilter]);

  useEffect(() => { fetchTimeline(); }, [fetchTimeline]);

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Time range */}
        <div className="flex rounded-lg border border-edge overflow-hidden">
          {([6, 24, 72, 168] as const).map((h) => (
            <button
              key={h}
              onClick={() => setHours(h)}
              className={`px-3 py-1.5 text-[12px] font-medium transition-colors ${
                hours === h ? "bg-gold/10 text-gold" : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
              }`}
            >
              {h < 24 ? `${h}h` : `${h / 24}d`}
            </button>
          ))}
        </div>

        {/* Source filter */}
        <div className="flex rounded-lg border border-edge overflow-hidden">
          <button
            onClick={() => setSourceFilter("all")}
            className={`px-3 py-1.5 text-[12px] font-medium transition-colors ${
              sourceFilter === "all" ? "bg-gold/10 text-gold" : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
            }`}
          >
            All
          </button>
          {(Object.keys(SOURCE_STYLES) as TimelineEventSource[]).map((src) => (
            <button
              key={src}
              onClick={() => setSourceFilter(src)}
              className={`px-3 py-1.5 text-[12px] font-medium transition-colors ${
                sourceFilter === src ? "bg-gold/10 text-gold" : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
              }`}
            >
              {SOURCE_STYLES[src].label}
            </button>
          ))}
        </div>

        {/* Event count */}
        {data && (
          <span className="text-[12px] text-ink-faint ml-auto">
            {data.total} event{data.total !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Loading */}
      {loading && !data && (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="flex gap-3">
              <div className="skeleton w-6 h-6 rounded-full" />
              <div className="flex-1 space-y-2">
                <div className="skeleton h-4 w-3/4 rounded" />
                <div className="skeleton h-3 w-1/2 rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Events */}
      {data && data.events.length === 0 && (
        <div className="text-center py-12 text-[13px] text-ink-faint">
          No events in the last {hours < 24 ? `${hours} hours` : `${hours / 24} days`}
        </div>
      )}

      {data && data.events.length > 0 && (
        <div className="pl-1">
          {data.events.map((event) => (
            <EventRow key={event.id} event={event} onNavigate={onNavigate} />
          ))}
          {data.hasMore && (
            <div className="text-center pt-2">
              <span className="text-[12px] text-ink-faint">
                {data.total - data.events.length} more events
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

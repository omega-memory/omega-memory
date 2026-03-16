import { useMemo, useState } from "react";
import type {
  CoordinationSession,
  CoordinationMessage,
  CoordinationHandoff,
  CoordinationTask,
  CoordinationGitEvent,
} from "../lib/types";
import { enrichProject, type Freshness, getFreshness } from "../lib/coordination-utils";

const BAR_COLORS: Record<Freshness, { bg: string; border: string; text: string }> = {
  active: { bg: "bg-emerald-500/30", border: "border-emerald-400/50", text: "text-emerald-400" },
  idle: { bg: "bg-amber-500/20", border: "border-amber-400/40", text: "text-amber-400" },
  stale: { bg: "bg-red-500/15", border: "border-red-400/30", text: "text-red-400" },
};

// ── Helpers ───────────────────────────────────────────────────────────

function formatDuration(ms: number): string {
  const totalMin = Math.floor(ms / 60_000);
  if (totalMin < 1) return "<1m";
  if (totalMin < 60) return `${totalMin}m`;
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Event types ──────────────────────────────────────────────────────

type TimelineEvent = {
  timestamp: number;
  type: "msg_sent" | "msg_received" | "handoff" | "git_commit" | "task_created" | "task_completed";
  label: string;
};

const EVENT_STYLES: Record<TimelineEvent["type"], { color: string; shape: string }> = {
  msg_sent:       { color: "text-amber-400",   shape: "M0,6 L4,0 L8,6Z" },          // upward triangle
  msg_received:   { color: "text-blue-400",    shape: "M0,0 L4,6 L8,0Z" },          // downward triangle
  handoff:        { color: "text-orange-400",   shape: "M4,0 L8,4 L4,8 L0,4Z" },    // diamond
  git_commit:     { color: "text-emerald-400",  shape: "" },                          // circle (uses <circle>)
  task_created:   { color: "text-gold",         shape: "M4,0 V8 M0,4 H8" },          // plus (uses <path>)
  task_completed: { color: "text-emerald-400",  shape: "M1,4 L3,6 L7,2" },           // checkmark
};

// ── Component ────────────────────────────────────────────────────────

interface TimelineStripProps {
  sessions: CoordinationSession[];
  messages?: CoordinationMessage[];
  handoffs?: CoordinationHandoff[];
  tasks?: CoordinationTask[];
  gitEvents?: CoordinationGitEvent[];
  onSessionClick?: (sessionId: string) => void;
  playbackTime?: number | null;
  onScrub?: (timestamp: number) => void;
  isPlaying?: boolean;
}

export default function TimelineStrip({
  sessions,
  messages = [],
  handoffs = [],
  tasks = [],
  gitEvents = [],
  onSessionClick,
  playbackTime,
  onScrub,
  isPlaying,
}: TimelineStripProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [hoveredEvent, setHoveredEvent] = useState<{ x: number; y: number; label: string } | null>(null);

  // Build session events map
  const sessionEvents = useMemo(() => {
    const map = new Map<string, TimelineEvent[]>();

    for (const msg of messages) {
      if (msg.from_session) {
        const list = map.get(msg.from_session) ?? [];
        list.push({ timestamp: new Date(msg.created_at).getTime(), type: "msg_sent", label: `Sent: ${msg.subject}` });
        map.set(msg.from_session, list);
      }
      if (msg.to_session) {
        const list = map.get(msg.to_session) ?? [];
        list.push({ timestamp: new Date(msg.created_at).getTime(), type: "msg_received", label: `Received: ${msg.subject}` });
        map.set(msg.to_session, list);
      }
    }

    for (const h of handoffs) {
      const list = map.get(h.session_id) ?? [];
      list.push({
        timestamp: new Date(h.created_at).getTime(),
        type: "handoff",
        label: `Handoff: ${h.key_context?.slice(0, 40) || "session end"}`,
      });
      map.set(h.session_id, list);
    }

    // Tasks lack session_id in the frontend CoordinationTask type (the DB has it,
    // but the API/type doesn't expose it). Group all task events under a synthetic
    // "__system__" lane so they still appear on the timeline rather than being lost.
    for (const t of tasks) {
      const systemList = map.get("__system__") ?? [];
      systemList.push({
        timestamp: new Date(t.created_at).getTime(),
        type: "task_created",
        label: `Task: ${t.title?.slice(0, 40) || "untitled"}`,
      });
      if (t.completed_at) {
        systemList.push({
          timestamp: new Date(t.completed_at).getTime(),
          type: "task_completed",
          label: `Done: ${t.title?.slice(0, 40) || "untitled"}`,
        });
      }
      map.set("__system__", systemList);
    }

    for (const ge of gitEvents) {
      if (!ge.session_id) continue;
      const list = map.get(ge.session_id) ?? [];
      if (ge.event_type === "commit") {
        list.push({
          timestamp: new Date(ge.created_at).getTime(),
          type: "git_commit",
          label: `Commit: ${ge.commit_hash?.slice(0, 7) || ""} ${ge.message?.slice(0, 30) || ""}`,
        });
      }
      map.set(ge.session_id, list);
    }

    return map;
  }, [messages, handoffs, tasks, gitEvents]);

  const { timeRange, rows } = useMemo(() => {
    if (sessions.length === 0) return { timeRange: { start: 0, end: 0, span: 1 }, rows: [] };

    const now = Date.now();

    // Find the earliest start and latest heartbeat
    let earliest = now;
    let latest = now;
    for (const s of sessions) {
      const start = new Date(s.started_at).getTime();
      const end = new Date(s.last_heartbeat).getTime();
      if (start < earliest) earliest = start;
      if (end > latest) latest = end;
    }

    // Pad 5% on each side for breathing room, minimum 5 minutes span
    const rawSpan = Math.max(latest - earliest, 5 * 60_000);
    const pad = rawSpan * 0.05;
    const start = earliest - pad;
    const end = Math.max(latest + pad, now + pad);
    const span = end - start;

    // Group by project (enriched names), sort by start time
    const grouped = new Map<string, CoordinationSession[]>();
    for (const s of sessions) {
      const { name } = enrichProject(s.project, undefined, s.task);
      const group = grouped.get(name) ?? [];
      group.push(s);
      grouped.set(name, group);
    }

    // Build rows: one per session, grouped under project labels
    const rows: {
      type: "label" | "bar";
      label?: string;
      session?: CoordinationSession;
      freshness?: Freshness;
      leftPct?: number;
      widthPct?: number;
      duration?: number;
      events?: TimelineEvent[];
    }[] = [];

    for (const [proj, group] of grouped) {
      rows.push({ type: "label", label: proj });
      // Sort within group: most recent first
      const sorted = [...group].sort(
        (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime(),
      );
      for (const s of sorted) {
        const sStart = new Date(s.started_at).getTime();
        const sEnd = new Date(s.last_heartbeat).getTime();
        const freshness = getFreshness(s.last_heartbeat);
        const leftPct = ((sStart - start) / span) * 100;
        const widthPct = Math.max(((sEnd - sStart) / span) * 100, 0.5); // min 0.5% for visibility
        rows.push({
          type: "bar",
          session: s,
          freshness,
          leftPct,
          widthPct,
          duration: sEnd - sStart,
          events: sessionEvents.get(s.session_id) ?? [],
        });
      }
    }

    return {
      timeRange: { start, end, span },
      rows,
    };
  }, [sessions, sessionEvents]);

  if (sessions.length === 0) return null;

  const now = Date.now();
  const nowPct = ((now - timeRange.start) / timeRange.span) * 100;

  // Generate time markers (every ~15-30 minutes)
  const markerInterval = timeRange.span > 3 * 60 * 60_000 ? 60 * 60_000 : // >3h: hourly
    timeRange.span > 60 * 60_000 ? 30 * 60_000 : // >1h: 30min
    15 * 60_000; // else: 15min

  const markers: { pct: number; label: string }[] = [];
  const firstMarker = Math.ceil(timeRange.start / markerInterval) * markerInterval;
  for (let t = firstMarker; t <= timeRange.end; t += markerInterval) {
    const pct = ((t - timeRange.start) / timeRange.span) * 100;
    if (pct >= 2 && pct <= 98) {
      markers.push({ pct, label: formatTime(new Date(t).toISOString()) });
    }
  }

  return (
    <div className="absolute bottom-0 left-0 right-0 z-10 bg-canvas/90 backdrop-blur-md border-t border-edge/40">
      {/* Toggle handle */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        className="w-full flex items-center justify-center gap-2 py-1 hover:bg-surface-hover/50 transition-colors group"
      >
        <svg
          className={`w-3.5 h-3.5 text-ink-faint group-hover:text-ink-secondary transition-all ${collapsed ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
        <span className="text-[10px] text-ink-faint group-hover:text-ink-secondary uppercase tracking-wider">
          Timeline
        </span>
      </button>

      {!collapsed && (
        <div className="px-4 pb-3">
          {/* Time axis with markers */}
          <div className="relative h-4 mb-1">
            {markers.map((m, i) => (
              <span
                key={i}
                className="absolute text-[9px] text-ink-faint font-mono -translate-x-1/2"
                style={{ left: `${m.pct}%` }}
              >
                {m.label}
              </span>
            ))}
          </div>

          {/* Timeline body */}
          <div
            className="relative"
            onClick={(e) => {
              if (!onScrub) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - rect.left) / rect.width;
              const ts = timeRange.start + pct * timeRange.span;
              onScrub(Math.round(ts));
            }}
          >
            {/* Now line */}
            {nowPct >= 0 && nowPct <= 100 && (
              <div
                className="absolute top-0 bottom-0 w-px bg-emerald-400/60 z-10"
                style={{ left: `${nowPct}%` }}
              >
                <div className="absolute -top-[2px] left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-emerald-400" />
              </div>
            )}

            {/* Playback cursor (gold) */}
            {playbackTime != null && (() => {
              const pct = ((playbackTime - timeRange.start) / timeRange.span) * 100;
              if (pct < 0 || pct > 100) return null;
              return (
                <div
                  className="absolute top-0 bottom-0 w-0.5 z-20 cursor-col-resize"
                  style={{ left: `${pct}%`, backgroundColor: "rgba(234,179,8,0.8)" }}
                >
                  <div className={`absolute -top-[2px] left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-yellow-400 shadow-md shadow-yellow-400/40 ${isPlaying ? "animate-pulse" : ""}`} />
                </div>
              );
            })()}

            {/* Floating tooltip for hovered events */}
            {hoveredEvent && (
              <div
                className="absolute z-30 pointer-events-none bg-surface-elevated border border-edge px-2 py-1 rounded-md shadow-lg whitespace-nowrap"
                style={{ left: hoveredEvent.x, top: hoveredEvent.y - 28 }}
              >
                <span className="text-[9px] text-ink-secondary">{hoveredEvent.label}</span>
              </div>
            )}

            {/* Rows */}
            <div className="space-y-0.5 max-h-[40vh] overflow-y-auto">
              {rows.map((row, i) => {
                if (row.type === "label") {
                  return (
                    <div key={`label-${i}`} className="flex items-center gap-2 py-1">
                      <span className="text-[10px] font-medium text-ink-tertiary uppercase tracking-wider">
                        {row.label}
                      </span>
                      <div className="flex-1 h-px bg-edge/30" />
                    </div>
                  );
                }

                const s = row.session!;
                const colors = BAR_COLORS[row.freshness!];
                const isActive = row.freshness === "active";
                const events = row.events ?? [];
                const useDensityMode = events.length > 10;

                return (
                  <div
                    key={s.session_id}
                    className="relative h-6 group cursor-pointer"
                    onClick={() => onSessionClick?.(s.session_id)}
                  >
                    {/* Session bar */}
                    <div
                      className={`absolute top-0.5 h-5 rounded-sm border ${colors.bg} ${colors.border} transition-all group-hover:brightness-125`}
                      style={{
                        left: `${row.leftPct}%`,
                        width: `${row.widthPct}%`,
                        minWidth: "4px",
                      }}
                    >
                      {/* Pulse on active bars */}
                      {isActive && (
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1">
                          <div className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-75" />
                        </div>
                      )}

                      {/* Event markers */}
                      {!useDensityMode && events.length > 0 && (
                        <div className="absolute inset-0 overflow-hidden">
                          {events.map((evt, ei) => {
                            const barStart = new Date(s.started_at).getTime();
                            const barEnd = new Date(s.last_heartbeat).getTime();
                            const barSpan = barEnd - barStart;
                            if (barSpan <= 0) return null;
                            const pct = ((evt.timestamp - barStart) / barSpan) * 100;
                            if (pct < 0 || pct > 100) return null;
                            const style = EVENT_STYLES[evt.type];
                            return (
                              <svg
                                key={ei}
                                className={`absolute top-1/2 -translate-y-1/2 w-2 h-2 ${style.color}`}
                                style={{ left: `${pct}%`, marginLeft: "-4px" }}
                                viewBox="0 0 8 8"
                                fill="currentColor"
                                stroke="none"
                                onMouseEnter={(e) => setHoveredEvent({
                                  x: e.clientX,
                                  y: e.clientY,
                                  label: evt.label,
                                })}
                                onMouseLeave={() => setHoveredEvent(null)}
                              >
                                {evt.type === "git_commit" ? (
                                  <circle cx="4" cy="4" r="3" />
                                ) : evt.type === "task_created" || evt.type === "task_completed" ? (
                                  <path d={style.shape} fill="none" stroke="currentColor" strokeWidth="1.5" />
                                ) : (
                                  <path d={style.shape} />
                                )}
                              </svg>
                            );
                          })}
                        </div>
                      )}

                      {/* Density mode: heatmap gradient for >10 events */}
                      {useDensityMode && (
                        <div
                          className="absolute inset-0 rounded-sm opacity-50"
                          style={{
                            background: (() => {
                              // Create 10 buckets across the bar
                              const barStart = new Date(s.started_at).getTime();
                              const barEnd = new Date(s.last_heartbeat).getTime();
                              const barSpan = barEnd - barStart;
                              if (barSpan <= 0) return "transparent";
                              const buckets = new Array(10).fill(0);
                              for (const evt of events) {
                                const idx = Math.min(Math.floor(((evt.timestamp - barStart) / barSpan) * 10), 9);
                                if (idx >= 0) buckets[idx]++;
                              }
                              const max = Math.max(...buckets, 1);
                              const stops = buckets.map((count, i) => {
                                const pct = (i / 9) * 100;
                                const alpha = (count / max) * 0.6;
                                return `rgba(251,191,36,${alpha}) ${pct}%`;
                              });
                              return `linear-gradient(to right, ${stops.join(", ")})`;
                            })(),
                          }}
                        />
                      )}
                    </div>

                    {/* Tooltip on hover */}
                    <div
                      className="absolute top-0.5 h-5 flex items-center pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity z-20"
                      style={{
                        left: `${Math.min(row.leftPct! + row.widthPct! + 0.5, 85)}%`,
                      }}
                    >
                      <div className="bg-surface-elevated border border-edge px-2 py-0.5 rounded-md shadow-lg whitespace-nowrap">
                        <span className={`text-[10px] font-mono ${colors.text}`}>
                          {s.session_id.slice(0, 8)}
                        </span>
                        <span className="text-[10px] text-ink-faint ml-2">
                          {formatTime(s.started_at)} - {formatTime(s.last_heartbeat)}
                        </span>
                        <span className="text-[10px] text-ink-tertiary ml-2">
                          {formatDuration(row.duration!)}
                        </span>
                        {events.length > 0 && (
                          <span className="text-[10px] text-amber-400/70 ml-2">
                            {events.length} event{events.length !== 1 ? "s" : ""}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

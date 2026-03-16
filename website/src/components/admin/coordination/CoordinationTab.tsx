import React, { useState, useMemo, useEffect, useCallback, useRef, Suspense, lazy } from "react";
import { useCoordinationData } from "./useCoordinationData";
import TimelineStrip from "./TimelineStrip";
import { buildReplayTimeline, findEventIndex, filterAtTime } from "./replay-utils";
import ReplayControls from "./ReplayControls";
import EnforcementFeed from "./EnforcementFeed";
import type {
  CoordinationSession, CoordinationFileClaim, CoordinationFileRead,
  CoordinationAuditEntry, CoordinationTask, CoordinationMessage,
  CoordinationHandoff, CoordinationGitEvent, CoordinationIntent,
  CoordinationDecision,
} from "../lib/types";

import { useProjects } from "../hooks/useProjects";
import {
  type Freshness,
  STALE_THRESHOLD_MS,
  getFreshness,
  formatUptime,
  formatTimeAgo,
  enrichProject,
} from "../lib/coordination-utils";

const CoordinationFlow = lazy(() => import("./CoordinationFlow"));

// ── Helpers ──────────────────────────────────────────────────────────

const FRESHNESS_STYLE: Record<Freshness, { dot: string; label: string; badge: string }> = {
  active: { dot: "bg-emerald-400", label: "Active", badge: "bg-emerald-400/10 text-emerald-400 border-emerald-400/20" },
  idle: { dot: "bg-amber-400", label: "Idle", badge: "bg-amber-400/10 text-amber-400 border-amber-400/20" },
  stale: { dot: "bg-red-400", label: "Stale", badge: "bg-red-400/10 text-red-400 border-red-400/20" },
};

// ── Bar dropdown ─────────────────────────────────────────────────────

function BarDropdown({
  label,
  active,
  activeColor,
  children,
}: {
  label: string;
  active?: boolean;
  activeColor?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`text-[11px] px-2.5 py-1.5 rounded-lg border backdrop-blur-sm transition-colors flex items-center gap-1 ${
          active
            ? `bg-canvas/80 ${activeColor ?? "text-ink-secondary"} border-edge/40`
            : open
              ? "bg-canvas/80 text-ink-secondary border-edge/40"
              : "bg-canvas/80 text-ink-faint border-edge/40 hover:text-ink-secondary"
        }`}
      >
        {label}
        <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
        </svg>
      </button>
      {open && (
        <div className="absolute top-full right-0 mt-1 min-w-[140px] py-1 rounded-lg bg-surface-elevated border border-edge shadow-xl shadow-black/40 z-20">
          {children}
        </div>
      )}
    </div>
  );
}

// ── Time range presets ───────────────────────────────────────────────

type TimeRange = "live" | "24h" | "7d" | "30d";

const TIME_RANGE_LABELS: Record<TimeRange, string> = {
  live: "Live",
  "24h": "24h",
  "7d": "7 days",
  "30d": "30 days",
};

function getTimeRangeDates(range: TimeRange): { since: string | null; until: string | null } {
  if (range === "live") return { since: null, until: null };
  const now = new Date();
  const hours = range === "24h" ? 24 : range === "7d" ? 7 * 24 : 30 * 24;
  const since = new Date(now.getTime() - hours * 60 * 60_000).toISOString();
  return { since, until: now.toISOString() };
}

// ── Memory types → display ──────────────────────────────────────────

interface SessionMemory {
  id: string;
  content: string;
  event_type: string;
  created_at: string;
  project: string | null;
  memory_type: string | null;
  priority: number | null;
  access_count: number | null;
}

const MEMORY_TYPE_STYLE: Record<string, { icon: string; color: string }> = {
  decision: { icon: "D", color: "text-type-decision bg-type-decision/10 border-type-decision/20" },
  lesson: { icon: "L", color: "text-type-lesson bg-type-lesson/10 border-type-lesson/20" },
  preference: { icon: "P", color: "text-type-preference bg-type-preference/10 border-type-preference/20" },
  fact: { icon: "F", color: "text-[#6b9fff] bg-[#6b9fff]/10 border-[#6b9fff]/20" },
  session: { icon: "S", color: "text-type-session bg-type-session/10 border-type-session/20" },
  task: { icon: "T", color: "text-type-task bg-type-task/10 border-type-task/20" },
};

function getMemoryStyle(type: string) {
  return MEMORY_TYPE_STYLE[type] ?? { icon: type[0]?.toUpperCase() ?? "?", color: "text-ink-faint bg-ink-faint/10 border-ink-faint/20" };
}

// ── Detail Panel ─────────────────────────────────────────────────────

function SessionDetail({
  session,
  claims,
  onClose,
  onDismiss,
}: {
  session: CoordinationSession;
  claims: CoordinationFileClaim[];
  onClose: () => void;
  onDismiss?: (sessionId: string) => void;
}) {
  const { resolvePathToProject } = useProjects();
  const freshness = getFreshness(session.last_heartbeat);
  const style = FRESHNESS_STYLE[freshness];
  const sessionClaims = claims.filter((c) => c.session_id === session.session_id);
  const filePaths = sessionClaims.map((c) => c.file_path);
  const enriched = enrichProject(session.project, filePaths, session.task, resolvePathToProject);
  const uptime = formatUptime(session.started_at);
  const heartbeatAgo = formatTimeAgo(session.last_heartbeat);

  // Fetch all session detail data
  const [memories, setMemories] = useState<SessionMemory[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(true);
  const [audit, setAudit] = useState<CoordinationAuditEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(true);
  const [tasks, setTasks] = useState<CoordinationTask[]>([]);
  const [tasksLoading, setTasksLoading] = useState(true);
  const [messages, setMessages] = useState<CoordinationMessage[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(true);
  const [handoffs, setHandoffs] = useState<CoordinationHandoff[]>([]);
  const [handoffsLoading, setHandoffsLoading] = useState(true);
  const [gitEvents, setGitEvents] = useState<CoordinationGitEvent[]>([]);
  const [gitEventsLoading, setGitEventsLoading] = useState(true);
  const [intents, setIntents] = useState<CoordinationIntent[]>([]);
  const [intentsLoading, setIntentsLoading] = useState(true);
  const [decisions, setDecisions] = useState<CoordinationDecision[]>([]);
  const [decisionsLoading, setDecisionsLoading] = useState(true);
  const [entities, setEntities] = useState<{ id: string; name: string; entityType: string; memoryCount: number }[]>([]);
  const [entitiesLoading, setEntitiesLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const sid = encodeURIComponent(session.session_id);

    const fetchEndpoint = <T,>(url: string, setter: (d: T[]) => void, loadSetter: (b: boolean) => void, key: string) => {
      loadSetter(true);
      fetch(url)
        .then((res) => res.json())
        .then((data) => { if (!cancelled) { setter(data[key] ?? []); loadSetter(false); } })
        .catch(() => { if (!cancelled) loadSetter(false); });
    };

    fetchEndpoint(`/api/admin/coordination/memories?session_id=${sid}`, setMemories, setMemoriesLoading, "memories");
    fetchEndpoint(`/api/admin/coordination/audit?session_id=${sid}`, setAudit, setAuditLoading, "audit");
    fetchEndpoint(`/api/admin/coordination/tasks?session_id=${sid}`, setTasks, setTasksLoading, "tasks");
    fetchEndpoint(`/api/admin/coordination/messages?session_id=${sid}`, setMessages, setMessagesLoading, "messages");
    fetchEndpoint(`/api/admin/coordination/handoffs?session_id=${sid}`, setHandoffs, setHandoffsLoading, "handoffs");
    fetchEndpoint(`/api/admin/coordination/git-events?session_id=${sid}`, setGitEvents, setGitEventsLoading, "git_events");
    fetchEndpoint(`/api/admin/coordination/intents?session_id=${sid}`, setIntents, setIntentsLoading, "intents");
    fetchEndpoint(`/api/admin/coordination/decisions?session_id=${sid}`, setDecisions, setDecisionsLoading, "decisions");

    // Fetch entities linked to this session's memories
    setEntitiesLoading(true);
    fetch(`/api/admin/coordination/entities?session_id=${sid}`)
      .then((res) => res.json())
      .then((data) => { if (!cancelled) { setEntities(data.entities ?? []); setEntitiesLoading(false); } })
      .catch(() => { if (!cancelled) setEntitiesLoading(false); });

    return () => { cancelled = true; };
  }, [session.session_id]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 z-10"
        onClick={onClose}
      />
      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 w-[340px] bg-surface-elevated border-l border-edge z-20 flex flex-col shadow-2xl shadow-black/30 animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-edge">
          <div className="flex items-center gap-2 min-w-0">
            <span className={`w-2 h-2 rounded-full ${style.dot} shrink-0`} />
            <div className="min-w-0">
              <span className="text-[13px] font-semibold text-ink truncate block">{enriched.name}</span>
              {enriched.summary && (
                <span className="text-[10px] text-ink-faint truncate block">{enriched.summary}</span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            {onDismiss && (
              <button
                onClick={() => onDismiss(session.session_id)}
                className="h-7 px-2 flex items-center gap-1 rounded-md text-[10px] font-medium text-red-400 hover:bg-red-500/10 transition-colors"
                title="Dismiss session"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                </svg>
                Dismiss
              </button>
            )}
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
          {/* Status badge */}
          <div className="flex items-center gap-2">
            <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full border ${style.badge}`}>
              {style.label}
            </span>
            <span className="text-[11px] text-ink-faint">
              {uptime} uptime
            </span>
          </div>

          {/* Task */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Current Task</div>
            <div className="text-[13px] text-ink-secondary leading-relaxed">
              {session.task || "No task reported"}
            </div>
          </div>

          {/* Info grid */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Last Heartbeat</div>
              <div className="text-[12px] text-ink-secondary">{heartbeatAgo}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Started</div>
              <div className="text-[12px] text-ink-secondary">
                {new Date(session.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Project Path</div>
              <div className="text-[12px] text-ink-secondary font-mono truncate" title={session.project}>
                {session.project || "N/A"}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Session</div>
              <div className="text-[12px] text-ink-secondary font-mono">{session.session_id.slice(0, 12)}</div>
            </div>
          </div>

          {/* Tasks */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Tasks {!tasksLoading && tasks.length > 0 && `(${tasks.length})`}
            </div>
            {tasksLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading tasks...</span>
              </div>
            ) : tasks.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No tasks</div>
            ) : (
              <div className="space-y-1.5">
                {tasks.map((t) => {
                  const statusStyle = t.status === "completed"
                    ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
                    : t.status === "in_progress"
                    ? "text-amber-400 bg-amber-400/10 border-amber-400/20"
                    : "text-ink-faint bg-ink-faint/10 border-ink-faint/20";
                  const statusLabel = t.status === "in_progress" ? "WIP" : t.status;
                  return (
                    <div key={t.id} className="px-2.5 py-2 rounded-lg bg-surface">
                      <div className="flex items-center gap-2">
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${statusStyle}`}>
                          {statusLabel}
                        </span>
                        <span className="text-[11px] text-ink-secondary truncate flex-1">{t.title}</span>
                      </div>
                      {t.status === "in_progress" && t.progress > 0 && (
                        <div className="mt-1.5 h-1 rounded-full bg-surface-hover overflow-hidden">
                          <div
                            className="h-full rounded-full bg-amber-400/60 transition-all"
                            style={{ width: `${t.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Tool Activity */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Tool Activity {!auditLoading && audit.length > 0 && `(${audit.length})`}
            </div>
            {auditLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading activity...</span>
              </div>
            ) : audit.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No tool calls recorded</div>
            ) : (
              <div className="max-h-[200px] overflow-y-auto space-y-0.5">
                {[...audit].reverse().map((a) => {
                  const isError = a.result_status !== "ok";
                  const isSlow = a.latency_ms != null && a.latency_ms > 5000;
                  return (
                    <div
                      key={a.id}
                      className={`flex items-center gap-2 px-2 py-1 rounded text-[10px] font-mono ${
                        isError ? "text-red-400 bg-red-500/[0.06]" : "text-ink-tertiary"
                      }`}
                    >
                      {a.call_index != null && (
                        <span className="text-ink-faint w-5 text-right shrink-0">#{a.call_index}</span>
                      )}
                      <span className="truncate flex-1">{a.tool_name}</span>
                      {a.latency_ms != null && (
                        <span className={`shrink-0 tabular-nums ${isSlow ? "text-amber-400" : "text-ink-faint"}`}>
                          {a.latency_ms >= 1000 ? `${(a.latency_ms / 1000).toFixed(1)}s` : `${a.latency_ms}ms`}
                        </span>
                      )}
                      <span className="text-ink-faint shrink-0">{formatTimeAgo(a.created_at)}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* File claims */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Files ({sessionClaims.length})
            </div>
            {sessionClaims.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No file claims</div>
            ) : (
              <div className="space-y-1.5">
                {sessionClaims.map((c) => {
                  const parts = c.file_path.split("/");
                  const short = parts.length > 3 ? `.../${parts.slice(-2).join("/")}` : c.file_path;
                  return (
                    <div
                      key={c.file_path}
                      className="flex items-start gap-2 px-2.5 py-1.5 rounded-lg bg-surface hover:bg-surface-hover transition-colors"
                    >
                      <svg className="w-3.5 h-3.5 text-ink-faint mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                      </svg>
                      <div className="min-w-0 flex-1">
                        <div className="text-[11px] font-mono text-ink-tertiary truncate" title={c.file_path}>
                          {short}
                        </div>
                        {c.task && (
                          <div className="text-[10px] text-ink-faint truncate mt-0.5">{c.task}</div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Memory feed */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Memory Feed {!memoriesLoading && memories.length > 0 && `(${memories.length})`}
            </div>
            {memoriesLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading memories...</span>
              </div>
            ) : memories.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No memories stored this session</div>
            ) : (
              <div className="space-y-1.5">
                {memories.map((m) => {
                  const mStyle = getMemoryStyle(m.event_type);
                  return (
                    <div
                      key={m.id}
                      className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-surface"
                    >
                      <span className={`shrink-0 w-5 h-5 rounded flex items-center justify-center text-[9px] font-bold border ${mStyle.color}`}>
                        {mStyle.icon}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="text-[11px] text-ink-secondary leading-relaxed line-clamp-3">
                          {m.content}
                        </div>
                        <div className="flex items-center gap-1.5 text-[9px] text-ink-faint mt-1">
                          <span>{m.event_type}</span>
                          {m.priority != null && m.priority > 1 && (
                            <span className="text-amber-400/70">P{m.priority}</span>
                          )}
                          {m.access_count != null && m.access_count > 0 && (
                            <span className="text-cyan-400/50">{m.access_count}x</span>
                          )}
                          <span>· {formatTimeAgo(m.created_at)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Messages */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Messages {!messagesLoading && messages.length > 0 && `(${messages.length})`}
            </div>
            {messagesLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading messages...</span>
              </div>
            ) : messages.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No messages</div>
            ) : (
              <div className="space-y-1.5 max-h-[180px] overflow-y-auto">
                {messages.map((msg) => {
                  const isSent = msg.from_session === session.session_id;
                  const typeColor = msg.msg_type === "request" ? "text-amber-400 bg-amber-400/10 border-amber-400/20"
                    : msg.msg_type === "reject" ? "text-red-400 bg-red-400/10 border-red-400/20"
                    : "text-blue-400 bg-blue-400/10 border-blue-400/20";
                  return (
                    <div key={msg.id} className="px-2.5 py-2 rounded-lg bg-surface">
                      <div className="flex items-center gap-1.5 mb-1">
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${typeColor}`}>
                          {msg.msg_type}
                        </span>
                        <span className="text-[9px] text-ink-faint">
                          {isSent ? "sent" : "received"}
                        </span>
                        {msg.read_at && (
                          <span className="text-[9px] text-emerald-400/50">read</span>
                        )}
                      </div>
                      <div className="text-[11px] text-ink-secondary font-medium truncate">{msg.subject}</div>
                      {msg.body && (
                        <div className="text-[10px] text-ink-tertiary line-clamp-2 mt-0.5">{msg.body}</div>
                      )}
                      <div className="text-[9px] text-ink-faint mt-1">{formatTimeAgo(msg.created_at)}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Git Events */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Git Activity {!gitEventsLoading && gitEvents.length > 0 && `(${gitEvents.length})`}
            </div>
            {gitEventsLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading git events...</span>
              </div>
            ) : gitEvents.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No git events</div>
            ) : (
              <div className="space-y-0.5 max-h-[160px] overflow-y-auto">
                {gitEvents.map((ge) => (
                  <div key={ge.id} className="flex items-center gap-2 px-2 py-1 rounded text-[10px] font-mono text-ink-tertiary">
                    <span className={`shrink-0 w-14 text-right ${
                      ge.event_type === "commit" ? "text-emerald-400" : ge.event_type === "push" ? "text-blue-400" : "text-ink-faint"
                    }`}>
                      {ge.event_type}
                    </span>
                    {ge.commit_hash && (
                      <span className="text-amber-400/70 shrink-0">{ge.commit_hash.slice(0, 7)}</span>
                    )}
                    <span className="truncate flex-1 text-ink-faint">{ge.message || ge.branch || ""}</span>
                    <span className="text-ink-faint shrink-0">{formatTimeAgo(ge.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Intents */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Intents {!intentsLoading && intents.length > 0 && `(${intents.length})`}
            </div>
            {intentsLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading intents...</span>
              </div>
            ) : intents.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No declared intents</div>
            ) : (
              <div className="space-y-1.5">
                {intents.map((intent) => {
                  const isExpired = intent.expires_at && new Date(intent.expires_at) < new Date();
                  const files: string[] = (() => { try { return JSON.parse(intent.target_files || "[]"); } catch { return []; } })();
                  return (
                    <div key={intent.id} className={`px-2.5 py-2 rounded-lg bg-surface ${isExpired ? "opacity-50" : ""}`}>
                      <div className="flex items-center gap-1.5 mb-1">
                        <span className="text-[9px] font-bold px-1.5 py-0.5 rounded border text-violet-400 bg-violet-400/10 border-violet-400/20">
                          {intent.intent_type}
                        </span>
                        {isExpired && <span className="text-[9px] text-red-400">expired</span>}
                        {intent.target_branch && (
                          <span className="text-[9px] font-mono text-cyan-400/70">{intent.target_branch}</span>
                        )}
                      </div>
                      <div className="text-[11px] text-ink-secondary line-clamp-2">{intent.description}</div>
                      {files.length > 0 && (
                        <div className="text-[9px] text-ink-faint mt-1">{files.length} target file{files.length !== 1 ? "s" : ""}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Decisions */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Decisions {!decisionsLoading && decisions.length > 0 && `(${decisions.length})`}
            </div>
            {decisionsLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading decisions...</span>
              </div>
            ) : decisions.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No decisions registered</div>
            ) : (
              <div className="space-y-1.5">
                {decisions.map((d) => {
                  const statusColor = d.status === "active" ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
                    : "text-ink-faint bg-ink-faint/10 border-ink-faint/20";
                  return (
                    <div key={d.id} className="px-2.5 py-2 rounded-lg bg-surface">
                      <div className="flex items-center gap-1.5 mb-1">
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${statusColor}`}>
                          {d.status}
                        </span>
                        <span className="text-[9px] font-mono text-amber-400/70 truncate">{d.domain}</span>
                      </div>
                      <div className="text-[11px] text-ink-secondary line-clamp-2">{d.decision}</div>
                      {d.rationale && (
                        <div className="text-[10px] text-ink-faint line-clamp-1 mt-0.5 italic">{d.rationale}</div>
                      )}
                      <div className="text-[9px] text-ink-faint mt-1">{formatTimeAgo(d.created_at)}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Handoffs */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Handoffs {!handoffsLoading && handoffs.length > 0 && `(${handoffs.length})`}
            </div>
            {handoffsLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading handoffs...</span>
              </div>
            ) : handoffs.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No handoffs</div>
            ) : (
              <div className="space-y-1.5">
                {handoffs.map((h) => {
                  const completed: string[] = (() => { try { return JSON.parse(h.completed_tasks || "[]"); } catch { return []; } })();
                  const blocked: string[] = (() => { try { return JSON.parse(h.blocked_items || "[]"); } catch { return []; } })();
                  const nextSteps: string[] = (() => { try { return JSON.parse(h.next_steps || "[]"); } catch { return []; } })();
                  const readBy: string[] = (() => { try { return JSON.parse(h.read_by || "[]"); } catch { return []; } })();
                  return (
                    <div key={h.id} className="px-2.5 py-2 rounded-lg bg-surface space-y-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] font-bold px-1.5 py-0.5 rounded border text-orange-400 bg-orange-400/10 border-orange-400/20">
                          HANDOFF
                        </span>
                        {h.git_branch && (
                          <span className="text-[9px] font-mono text-cyan-400/70">{h.git_branch}</span>
                        )}
                        <span className="text-[9px] text-ink-faint ml-auto">{formatTimeAgo(h.created_at)}</span>
                      </div>
                      {h.key_context && (
                        <div className="text-[11px] text-ink-secondary line-clamp-2">{h.key_context}</div>
                      )}
                      {completed.length > 0 && (
                        <div>
                          <span className="text-[9px] text-emerald-400/70">Completed ({completed.length}):</span>
                          {completed.slice(0, 3).map((t, i) => (
                            <div key={i} className="text-[10px] text-ink-tertiary truncate ml-2">- {t}</div>
                          ))}
                        </div>
                      )}
                      {blocked.length > 0 && (
                        <div>
                          <span className="text-[9px] text-red-400/70">Blocked ({blocked.length}):</span>
                          {blocked.slice(0, 2).map((b, i) => (
                            <div key={i} className="text-[10px] text-red-400/50 truncate ml-2">- {b}</div>
                          ))}
                        </div>
                      )}
                      {nextSteps.length > 0 && (
                        <div>
                          <span className="text-[9px] text-blue-400/70">Next ({nextSteps.length}):</span>
                          {nextSteps.slice(0, 2).map((n, i) => (
                            <div key={i} className="text-[10px] text-ink-faint truncate ml-2">- {n}</div>
                          ))}
                        </div>
                      )}
                      {readBy.length > 0 && (
                        <div className="text-[9px] text-ink-faint">Read by {readBy.length} session{readBy.length !== 1 ? "s" : ""}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Entities */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Entities {!entitiesLoading && entities.length > 0 && `(${entities.length})`}
            </div>
            {entitiesLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading entities...</span>
              </div>
            ) : entities.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No entities extracted</div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {entities.map((e) => {
                  const typeColors: Record<string, string> = {
                    person: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
                    project: "text-gold bg-gold/10 border-gold/20",
                    tool: "text-orange-400 bg-orange-400/10 border-orange-400/20",
                    technology: "text-sky-400 bg-sky-400/10 border-sky-400/20",
                    service: "text-pink-400 bg-pink-400/10 border-pink-400/20",
                    company: "text-blue-400 bg-blue-400/10 border-blue-400/20",
                    concept: "text-cyan-400 bg-cyan-400/10 border-cyan-400/20",
                  };
                  const color = typeColors[e.entityType] || "text-ink-tertiary bg-ink-faint/10 border-ink-faint/20";
                  return (
                    <span
                      key={e.id}
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${color}`}
                      title={`${e.entityType} - ${e.memoryCount} memories`}
                    >
                      {e.name}
                      {e.memoryCount > 1 && (
                        <span className="opacity-60">{e.memoryCount}</span>
                      )}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function FileDetail({
  claim,
  allClaims,
  sessions,
  onClose,
}: {
  claim: CoordinationFileClaim;
  allClaims: CoordinationFileClaim[];
  sessions: CoordinationSession[];
  onClose: () => void;
}) {
  const { resolvePathToProject } = useProjects();
  const displayProject = (raw: string | null | undefined): string => {
    if (!raw) return "Unknown";
    const match = resolvePathToProject(raw);
    if (match) return match.name;
    const segs = raw.replace(/\/+$/, "").split("/");
    const projIdx = segs.indexOf("Projects");
    if (projIdx >= 0 && segs[projIdx + 1]) return segs[projIdx + 1];
    return segs[segs.length - 1] || "Unknown";
  };

  const parts = claim.file_path.split("/");
  const filename = parts[parts.length - 1];
  const dir = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
  const ownerSession = sessions.find((s) => s.session_id === claim.session_id);

  // Find all agents claiming this same file path (conflict detection)
  const allClaimsForFile = allClaims.filter((c) => c.file_path === claim.file_path);
  const isConflict = allClaimsForFile.length > 1;

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 z-10"
        onClick={onClose}
      />
      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 w-[340px] bg-surface-elevated border-l border-edge z-20 flex flex-col shadow-2xl shadow-black/30 animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-edge">
          <div className="flex items-center gap-2 min-w-0">
            <svg className={`w-4 h-4 shrink-0 ${isConflict ? "text-red-400" : "text-ink-faint"}`} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
            <span className="text-[13px] font-semibold text-ink truncate">{filename}</span>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors shrink-0"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
          {/* Conflict banner */}
          {isConflict && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-red-500/[0.08] border border-red-500/20">
              <span className="text-red-400 text-[12px] mt-0.5 shrink-0">!</span>
              <div>
                <div className="text-[12px] font-medium text-red-400">File conflict</div>
                <div className="text-[11px] text-red-400/70 mt-0.5">
                  {allClaimsForFile.length} agents are working on this file simultaneously
                </div>
              </div>
            </div>
          )}

          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Full Path</div>
            <div className="text-[12px] font-mono text-ink-secondary break-all">{claim.file_path}</div>
          </div>

          {dir && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Directory</div>
              <div className="text-[12px] font-mono text-ink-tertiary break-all">{dir}</div>
            </div>
          )}

          {claim.task && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Task</div>
              <div className="text-[13px] text-ink-secondary">{claim.task}</div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Claimed</div>
              <div className="text-[12px] text-ink-secondary">{formatTimeAgo(claim.claimed_at)}</div>
            </div>
            {ownerSession && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">By Agent</div>
                <div className="text-[12px] text-ink-secondary">{displayProject(ownerSession.project)}</div>
              </div>
            )}
          </div>

          {/* Conflict details: show all claiming agents */}
          {isConflict && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
                Claiming Agents ({allClaimsForFile.length})
              </div>
              <div className="space-y-1.5">
                {allClaimsForFile.map((c) => {
                  const agent = sessions.find((s) => s.session_id === c.session_id);
                  const freshness = agent ? getFreshness(agent.last_heartbeat) : "stale";
                  const style = FRESHNESS_STYLE[freshness];
                  return (
                    <div
                      key={c.session_id}
                      className="flex items-center gap-2 px-2.5 py-2 rounded-lg bg-surface"
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${style.dot} shrink-0`} />
                      <div className="min-w-0 flex-1">
                        <div className="text-[12px] text-ink-secondary truncate">
                          {agent ? displayProject(agent.project) : c.session_id.slice(0, 8)}
                        </div>
                        {c.task && (
                          <div className="text-[10px] text-ink-faint truncate mt-0.5">{c.task}</div>
                        )}
                      </div>
                      <span className="text-[10px] text-ink-faint shrink-0">{formatTimeAgo(c.claimed_at)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Main Component ───────────────────────────────────────────────────

export default function CoordinationTab() {
  const [timeRange, setTimeRange] = useState<TimeRange>("live");
  const { since, until } = useMemo(() => getTimeRangeDates(timeRange), [timeRange]);
  const {
    sessions, fileClaims, fileReads,
    messages, handoffs, intents, decisions, tasks: coordTasks, gitEvents, metrics: coordMetrics,
    isLoading, isHistorical, error, refetch,
  } = useCoordinationData({ since, until });
  const [selectedNode, setSelectedNode] = useState<{
    type: "agent" | "file" | "inputFile";
    id: string;
  } | null>(null);
  const [showStale, setShowStale] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [sortBy, setSortBy] = useState<"default" | "conflicts" | "uptime" | "files">("default");
  const [showMessages, setShowMessages] = useState(true);
  const [showHandoffs, setShowHandoffs] = useState(true);
  const [showIntents, setShowIntents] = useState(true);
  const [dismissError, setDismissError] = useState<string | null>(null);

  // ── Replay state (only active when historical) ──
  const [playbackTime, setPlaybackTime] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const playbackRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(0);

  const dismissSessions = useCallback(async (sessionIds: string[]) => {
    if (sessionIds.length === 0) return;
    setCleaning(true);
    setDismissError(null);
    try {
      // Batch into chunks of 50 to avoid API rejection for large payloads
      const BATCH_SIZE = 50;
      const errors: string[] = [];
      for (let i = 0; i < sessionIds.length; i += BATCH_SIZE) {
        const chunk = sessionIds.slice(i, i + BATCH_SIZE);
        const res = await fetch("/api/admin/coordination/sessions", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_ids: chunk }),
        });
        if (!res.ok) {
          const text = await res.text().catch(() => res.statusText);
          errors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${text}`);
        }
      }
      if (errors.length > 0) {
        setDismissError(`Failed to clean some sessions: ${errors.join("; ")}`);
      } else {
        setSelectedNode(null);
      }
      refetch();
    } catch (err) {
      setDismissError(`Network error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCleaning(false);
    }
  }, [refetch]);

  // Build replay timeline from fetched data
  const replayTimeline = useMemo(
    () => isHistorical
      ? buildReplayTimeline(sessions, fileClaims, fileReads, messages, handoffs, intents, decisions, gitEvents)
      : [],
    [isHistorical, sessions, fileClaims, fileReads, messages, handoffs, intents, decisions, gitEvents],
  );

  // Reset replay state when timeRange changes
  useEffect(() => {
    setPlaybackTime(null);
    setIsPlaying(false);
    setPlaybackSpeed(1);
  }, [timeRange]);

  // Auto-initialize playbackTime to first event when entering historical mode
  // Without this, ReplayControls returns null when playbackTime is null,
  // so replay controls never appear until the user manually clicks "Replay".
  useEffect(() => {
    if (replayTimeline.length > 0 && playbackTime == null) {
      setPlaybackTime(replayTimeline[0].timestamp);
    }
  }, [replayTimeline, playbackTime]);

  // requestAnimationFrame playback loop
  useEffect(() => {
    if (!isPlaying || replayTimeline.length === 0 || playbackTime == null) {
      if (playbackRef.current) cancelAnimationFrame(playbackRef.current);
      playbackRef.current = null;
      return;
    }

    const lastTs = replayTimeline[replayTimeline.length - 1].timestamp;
    lastFrameRef.current = performance.now();

    function tick(now: number) {
      const delta = now - lastFrameRef.current;
      lastFrameRef.current = now;

      setPlaybackTime((prev) => {
        if (prev == null) return prev;
        const next = prev + delta * playbackSpeed;
        if (next >= lastTs) {
          setIsPlaying(false);
          return lastTs;
        }
        return next;
      });

      playbackRef.current = requestAnimationFrame(tick);
    }

    playbackRef.current = requestAnimationFrame(tick);
    return () => {
      if (playbackRef.current) cancelAnimationFrame(playbackRef.current);
    };
  }, [isPlaying, playbackSpeed, replayTimeline, playbackTime]);

  // Keyboard handler (Escape + replay controls)
  const closePanel = useCallback(() => setSelectedNode(null), []);
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") { closePanel(); return; }

      // Replay controls (only when historical and replay active)
      if (isHistorical && replayTimeline.length > 0 && playbackTime != null) {
        if (e.code === "Space") {
          e.preventDefault();
          setIsPlaying((v) => !v);
          return;
        }
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          const idx = findEventIndex(replayTimeline, playbackTime);
          if (idx > 0) { setPlaybackTime(replayTimeline[idx - 1].timestamp); setIsPlaying(false); }
          return;
        }
        if (e.key === "ArrowRight") {
          e.preventDefault();
          const idx = findEventIndex(replayTimeline, playbackTime);
          if (idx < replayTimeline.length - 1) { setPlaybackTime(replayTimeline[idx + 1].timestamp); setIsPlaying(false); }
          return;
        }
        if (e.key === "[") {
          const speeds = [1, 2, 5, 10];
          const i = speeds.indexOf(playbackSpeed);
          if (i > 0) setPlaybackSpeed(speeds[i - 1]);
          return;
        }
        if (e.key === "]") {
          const speeds = [1, 2, 5, 10];
          const i = speeds.indexOf(playbackSpeed);
          if (i < speeds.length - 1) setPlaybackSpeed(speeds[i + 1]);
          return;
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [closePanel, isHistorical, replayTimeline, playbackTime, playbackSpeed]);

  // Filter out stale sessions (>30min without heartbeat) unless toggled or in historical mode
  const visibleSessions = useMemo(() => {
    let filtered = isHistorical || showStale
      ? sessions
      : sessions.filter((s) => {
          const diff = Date.now() - new Date(s.last_heartbeat).getTime();
          return diff < STALE_THRESHOLD_MS;
        });
    if (playbackTime != null) {
      filtered = filtered.filter((s) => new Date(s.started_at).getTime() <= playbackTime);
    }
    return filtered;
  }, [sessions, showStale, isHistorical, playbackTime]);

  // File claims only for visible sessions
  const visibleClaims = useMemo(() => {
    const visibleIds = new Set(visibleSessions.map((s) => s.session_id));
    return fileClaims.filter((c) => visibleIds.has(c.session_id));
  }, [fileClaims, visibleSessions]);

  // File reads only for visible sessions
  const visibleReads = useMemo(() => {
    const visibleIds = new Set(visibleSessions.map((s) => s.session_id));
    return fileReads.filter((r) => visibleIds.has(r.session_id));
  }, [fileReads, visibleSessions]);

  // Messages involving visible sessions (including broadcasts where to_session is null)
  const visibleMessages = useMemo(() => {
    const visibleIds = new Set(visibleSessions.map((s) => s.session_id));
    return messages.filter((m) =>
      visibleIds.has(m.from_session) || (m.to_session && visibleIds.has(m.to_session)),
    );
  }, [messages, visibleSessions]);

  // Replay-filtered data (filters by timestamp for scrubbing)
  const replayFilteredClaims = useMemo(
    () => filterAtTime(visibleClaims, "claimed_at", playbackTime),
    [visibleClaims, playbackTime],
  );
  const replayFilteredReads = useMemo(
    () => filterAtTime(visibleReads, "first_read_at", playbackTime),
    [visibleReads, playbackTime],
  );
  const replayFilteredMessages = useMemo(
    () => filterAtTime(visibleMessages, "created_at", playbackTime),
    [visibleMessages, playbackTime],
  );
  const replayFilteredHandoffs = useMemo(
    () => filterAtTime(handoffs, "created_at", playbackTime),
    [handoffs, playbackTime],
  );
  const replayFilteredIntents = useMemo(
    () => filterAtTime(intents, "created_at", playbackTime),
    [intents, playbackTime],
  );
  const replayFilteredDecisions = useMemo(
    () => filterAtTime(decisions, "created_at", playbackTime),
    [decisions, playbackTime],
  );

  const staleCount = useMemo(
    () =>
      sessions.filter((s) => {
        const diff = Date.now() - new Date(s.last_heartbeat).getTime();
        return diff >= STALE_THRESHOLD_MS;
      }).length,
    [sessions],
  );



  const selectedSession = useMemo(
    () =>
      selectedNode?.type === "agent"
        ? sessions.find((s) => `agent-${s.session_id}` === selectedNode.id)
        : null,
    [selectedNode, sessions],
  );

  const selectedClaim = useMemo(
    () =>
      selectedNode?.type === "file"
        ? fileClaims.find(
            (f) => `file-${f.session_id}-${f.file_path}` === selectedNode.id,
          )
        : null,
    [selectedNode, fileClaims],
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center w-full h-full">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-gold/30 border-t-gold animate-spin" />
          <span className="text-[12px] text-ink-faint">Loading coordination data</span>
        </div>
      </div>
    );
  }

  const hasData = sessions.length > 0;

  return (
    <div className="relative w-full h-full">
      {/* Empty state */}
      {!hasData ? (
        <div className="flex flex-col items-center justify-center w-full h-full">
          <svg
            className="w-12 h-12 text-ink-faint mb-3"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"
            />
          </svg>
          <p className="text-[14px] text-ink-tertiary">
            {isHistorical ? "No sessions found for this period" : "No active agent sessions"}
          </p>
          <p className="text-[12px] text-ink-faint mt-1">
            {isHistorical
              ? "Try a different time range or switch back to Live view"
              : "Sessions appear here when agents register via OMEGA coordination"}
          </p>
          {isHistorical && (
            <div className="flex items-center gap-2 mt-4">
              {(["live", "24h", "7d", "30d"] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`text-[11px] px-2.5 py-1.5 rounded-lg border transition-colors ${
                    timeRange === range
                      ? range === "live"
                        ? "bg-emerald-400/10 text-emerald-400 border-emerald-400/20"
                        : "bg-blue-400/10 text-blue-400 border-blue-400/20"
                      : "bg-canvas/80 text-ink-faint border-edge/40 hover:text-ink-secondary"
                  }`}
                >
                  {TIME_RANGE_LABELS[range]}
                </button>
              ))}
            </div>
          )}
        </div>
      ) : (
        <>
          {/* Compact status bar */}
          <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
            {/* Left: live pulse + key counts */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-canvas/80 backdrop-blur-sm border border-edge/40 pointer-events-auto">
              {(() => {
                const hasActive = visibleSessions.some((s) => getFreshness(s.last_heartbeat) === "active");
                const dotColor = hasActive ? "bg-emerald-400" : "bg-amber-400";
                return (
                  <span className="relative flex h-1.5 w-1.5">
                    {hasActive && <span className={`absolute inset-0 rounded-full ${dotColor} animate-ping opacity-75`} />}
                    <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${dotColor}`} />
                  </span>
                );
              })()}
              {(() => {
                const subCount = visibleSessions.filter((s) => {
                  const meta = s.metadata as Record<string, unknown> | null;
                  return !!meta?.parent_session_id;
                }).length;
                const parentCount = visibleSessions.length - subCount;
                return subCount > 0 ? (
                  <>
                    <span className="text-[11px] font-mono font-semibold text-emerald-400">{parentCount}</span>
                    <span className="text-[10px] text-ink-faint">agents</span>
                    <span className="text-ink-faint/30">&middot;</span>
                    <span className="text-[11px] font-mono font-semibold text-sky-400">{subCount}</span>
                    <span className="text-[10px] text-sky-400/60">sub</span>
                  </>
                ) : (
                  <>
                    <span className="text-[11px] font-mono font-semibold text-emerald-400">{visibleSessions.length}</span>
                    <span className="text-[10px] text-ink-faint">agents</span>
                  </>
                );
              })()}
              {(() => {
                const uniqueFiles = new Set(replayFilteredClaims.map((f) => f.file_path)).size;
                return uniqueFiles > 0 ? (
                  <>
                    <span className="text-ink-faint/30">&middot;</span>
                    <span className="text-[11px] font-mono font-semibold text-ink-secondary">{uniqueFiles}</span>
                    <span className="text-[10px] text-ink-faint">files</span>
                  </>
                ) : null;
              })()}
              {(() => {
                const pathToSessions = new Map<string, Set<string>>();
                for (const f of replayFilteredClaims) {
                  const set = pathToSessions.get(f.file_path) ?? new Set();
                  set.add(f.session_id);
                  pathToSessions.set(f.file_path, set);
                }
                let conflicts = 0;
                for (const [, sids] of pathToSessions) { if (sids.size > 1) conflicts++; }
                return conflicts > 0 ? (
                  <>
                    <span className="text-ink-faint/30">&middot;</span>
                    <span className="text-[11px] font-mono font-semibold text-red-400">{conflicts}</span>
                    <span className="text-[10px] text-ink-faint">conflicts</span>
                  </>
                ) : null;
              })()}
              {(() => {
                const BLOCK_METRICS = new Set(["conflict_blocked_by_guard", "gate_check_medium", "gate_check_high", "deadlock_cycle"]);
                const blocks = coordMetrics ? coordMetrics.filter((m) => BLOCK_METRICS.has(m.metric_name)).length : 0;
                return blocks > 0 ? (
                  <>
                    <span className="text-ink-faint/30">&middot;</span>
                    <span className="text-[11px] font-mono font-semibold text-red-400">{blocks}</span>
                    <span className="text-[10px] text-ink-faint">blocks</span>
                  </>
                ) : null;
              })()}
              {playbackTime != null && (
                <>
                  <span className="text-ink-faint/30">&middot;</span>
                  <span className="text-[10px] font-mono text-gold">
                    {new Date(playbackTime).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </span>
                </>
              )}
            </div>

            {/* Right: dropdown controls */}
            <div className="flex items-center gap-1.5 pointer-events-auto">
              {/* Sort dropdown */}
              <BarDropdown
                label={sortBy === "default" ? "Sort" : sortBy.charAt(0).toUpperCase() + sortBy.slice(1)}
                active={sortBy !== "default"}
                activeColor="text-gold"
              >
                {(["default", "conflicts", "uptime", "files"] as const).map((key) => (
                  <button
                    key={key}
                    onClick={() => setSortBy(key)}
                    className={`w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors ${
                      sortBy === key ? "text-gold bg-gold/10" : "text-ink-secondary hover:bg-surface-hover"
                    }`}
                  >
                    {key === "default" ? "Default" : key.charAt(0).toUpperCase() + key.slice(1)}
                  </button>
                ))}
              </BarDropdown>

              {/* Layers dropdown */}
              <BarDropdown
                label="Layers"
                active={!showMessages || !showHandoffs || !showIntents}
                activeColor="text-blue-400"
              >
                <button
                  onClick={() => setShowMessages((v) => !v)}
                  className="w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors flex items-center justify-between"
                >
                  <span className={showMessages ? "text-blue-400" : "text-ink-faint"}>Messages</span>
                  {showMessages && <span className="text-[9px] text-blue-400">ON</span>}
                </button>
                <button
                  onClick={() => setShowHandoffs((v) => !v)}
                  className="w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors flex items-center justify-between"
                >
                  <span className={showHandoffs ? "text-orange-400" : "text-ink-faint"}>Handoffs</span>
                  {showHandoffs && <span className="text-[9px] text-orange-400">ON</span>}
                </button>
                <button
                  onClick={() => setShowIntents((v) => !v)}
                  className="w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors flex items-center justify-between"
                >
                  <span className={showIntents ? "text-violet-400" : "text-ink-faint"}>Intents</span>
                  {showIntents && <span className="text-[9px] text-violet-400">ON</span>}
                </button>
              </BarDropdown>

              {/* Time range dropdown */}
              <BarDropdown
                label={TIME_RANGE_LABELS[timeRange]}
                active={timeRange !== "live"}
                activeColor="text-blue-400"
              >
                {(["live", "24h", "7d", "30d"] as TimeRange[]).map((range) => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors ${
                      timeRange === range
                        ? range === "live" ? "text-emerald-400 bg-emerald-400/10" : "text-blue-400 bg-blue-400/10"
                        : "text-ink-secondary hover:bg-surface-hover"
                    }`}
                  >
                    {TIME_RANGE_LABELS[range]}
                  </button>
                ))}
                {isHistorical && replayTimeline.length > 0 && (
                  <>
                    <div className="border-t border-edge/40 my-1" />
                    <button
                      onClick={() => {
                        if (playbackTime != null) { setPlaybackTime(null); setIsPlaying(false); }
                        else { setPlaybackTime(replayTimeline[0].timestamp); }
                      }}
                      className={`w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors ${
                        playbackTime != null ? "text-gold bg-gold/10" : "text-ink-secondary hover:bg-surface-hover"
                      }`}
                    >
                      {playbackTime != null ? "Exit Replay" : `Replay (${replayTimeline.length} events)`}
                    </button>
                  </>
                )}
              </BarDropdown>

              {/* Stale badge + dropdown */}
              {staleCount > 0 && !isHistorical && (
                <BarDropdown
                  label={`+${staleCount} stale`}
                  active={showStale}
                  activeColor="text-red-400"
                >
                  <button
                    onClick={() => setShowStale((v) => !v)}
                    className="w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors flex items-center justify-between text-ink-secondary hover:bg-surface-hover"
                  >
                    <span>{showStale ? "Hide stale" : "Show stale"}</span>
                    {showStale && <span className="text-[9px] text-red-400">ON</span>}
                  </button>
                  <button
                    onClick={() => {
                      const staleIds = sessions
                        .filter((s) => Date.now() - new Date(s.last_heartbeat).getTime() >= STALE_THRESHOLD_MS)
                        .map((s) => s.session_id);
                      dismissSessions(staleIds);
                    }}
                    disabled={cleaning}
                    className="w-full text-left text-[11px] px-3 py-1.5 rounded transition-colors text-red-400 hover:bg-red-500/10 disabled:opacity-50"
                  >
                    {cleaning ? "Cleaning..." : "Clean all stale"}
                  </button>
                </BarDropdown>
              )}
            </div>
          </div>

          {/* Error banner + enforcement feed (stacked below status bar) */}
          <div className="absolute top-14 right-3 z-[5] flex flex-col items-end gap-2 pointer-events-none">
            {(error || dismissError) && (
              <div className="pointer-events-auto w-[calc(100vw-24px-var(--sidebar-width,200px))] max-w-[600px]">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/[0.08] backdrop-blur-sm border border-red-500/20">
                  <span className="text-[11px] text-red-400 flex-1">{error || dismissError}</span>
                  {dismissError && (
                    <button
                      onClick={() => setDismissError(null)}
                      className="text-red-400/70 hover:text-red-400 transition-colors"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            )}

            <div className="pointer-events-auto">
              <EnforcementFeed metrics={coordMetrics} />
            </div>
          </div>

          {/* Flow canvas */}
          <Suspense fallback={<div className="flex items-center justify-center h-full text-[13px] text-ink-tertiary">Loading coordination graph...</div>}>
            <CoordinationFlow
              sessions={visibleSessions}
              fileClaims={replayFilteredClaims}
              fileReads={replayFilteredReads}
              messages={replayFilteredMessages}
              handoffs={replayFilteredHandoffs}
              intents={replayFilteredIntents}
              decisions={replayFilteredDecisions}
              showMessages={showMessages}
              showHandoffs={showHandoffs}
              showIntents={showIntents}
              sortBy={sortBy}
              playbackTime={playbackTime}
              onNodeClick={(type, id) =>
                setSelectedNode((prev) =>
                  prev?.id === id ? null : { type, id },
                )
              }
            />
          </Suspense>

          {/* Replay controls (historical only, between flow and timeline) */}
          {isHistorical && (
            <ReplayControls
              timeline={replayTimeline}
              playbackTime={playbackTime}
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
              onSetPlaybackTime={setPlaybackTime}
              onSetPlaying={setIsPlaying}
              onSetSpeed={setPlaybackSpeed}
            />
          )}

          {/* Timeline strip (bottom) */}
          <TimelineStrip
            sessions={visibleSessions}
            messages={replayFilteredMessages}
            handoffs={replayFilteredHandoffs}
            tasks={coordTasks}
            gitEvents={gitEvents}
            playbackTime={playbackTime}
            isPlaying={isPlaying}
            onScrub={(ts) => {
              setPlaybackTime(ts);
              setIsPlaying(false);
            }}
            onSessionClick={(sessionId) =>
              setSelectedNode((prev) =>
                prev?.id === `agent-${sessionId}` ? null : { type: "agent", id: `agent-${sessionId}` },
              )
            }
          />

          {/* Detail slide-over */}
          {selectedSession && (
            <SessionDetail
              session={selectedSession}
              claims={fileClaims}
              onClose={closePanel}
              onDismiss={(sid) => dismissSessions([sid])}
            />
          )}
          {selectedClaim && (
            <FileDetail
              claim={selectedClaim}
              allClaims={fileClaims}
              sessions={sessions}
              onClose={closePanel}
            />
          )}
        </>
      )}
    </div>
  );
}

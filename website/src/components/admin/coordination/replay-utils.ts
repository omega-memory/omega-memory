import type {
  CoordinationSession, CoordinationFileClaim, CoordinationFileRead,
  CoordinationMessage, CoordinationHandoff, CoordinationIntent,
  CoordinationDecision, CoordinationGitEvent,
} from "../lib/types";

export type ReplayEventType =
  | "session_start" | "session_end"
  | "file_claim" | "file_read"
  | "message" | "handoff"
  | "intent" | "decision" | "git_event";

export interface ReplayEvent {
  timestamp: number;
  type: ReplayEventType;
  sessionId: string;
  data: CoordinationSession | CoordinationFileClaim | CoordinationFileRead
    | CoordinationMessage | CoordinationHandoff | CoordinationIntent
    | CoordinationDecision | CoordinationGitEvent;
}

export function buildReplayTimeline(
  sessions: CoordinationSession[],
  fileClaims: CoordinationFileClaim[],
  fileReads: CoordinationFileRead[],
  messages: CoordinationMessage[],
  handoffs: CoordinationHandoff[],
  intents: CoordinationIntent[],
  decisions: CoordinationDecision[],
  gitEvents: CoordinationGitEvent[],
): ReplayEvent[] {
  const events: ReplayEvent[] = [];

  for (const s of sessions) {
    events.push({
      timestamp: new Date(s.started_at).getTime(),
      type: "session_start",
      sessionId: s.session_id,
      data: s,
    });
    if (s.status === "ended" || s.status === "stopped") {
      events.push({
        timestamp: new Date(s.last_heartbeat).getTime(),
        type: "session_end",
        sessionId: s.session_id,
        data: s,
      });
    }
  }

  for (const f of fileClaims) {
    events.push({
      timestamp: new Date(f.claimed_at).getTime(),
      type: "file_claim",
      sessionId: f.session_id,
      data: f,
    });
  }

  for (const r of fileReads) {
    events.push({
      timestamp: new Date(r.first_read_at).getTime(),
      type: "file_read",
      sessionId: r.session_id,
      data: r,
    });
  }

  for (const m of messages) {
    events.push({
      timestamp: new Date(m.created_at).getTime(),
      type: "message",
      sessionId: m.from_session || "",
      data: m,
    });
  }

  for (const h of handoffs) {
    events.push({
      timestamp: new Date(h.created_at).getTime(),
      type: "handoff",
      sessionId: h.session_id,
      data: h,
    });
  }

  for (const i of intents) {
    events.push({
      timestamp: new Date(i.created_at).getTime(),
      type: "intent",
      sessionId: i.session_id,
      data: i,
    });
  }

  for (const d of decisions) {
    events.push({
      timestamp: new Date(d.created_at).getTime(),
      type: "decision",
      sessionId: d.decided_by,
      data: d,
    });
  }

  for (const g of gitEvents) {
    events.push({
      timestamp: new Date(g.created_at).getTime(),
      type: "git_event",
      sessionId: g.session_id || "",
      data: g,
    });
  }

  events.sort((a, b) => a.timestamp - b.timestamp);
  return events;
}

/** Find the index of the event nearest to (but not after) the given timestamp. */
export function findEventIndex(events: ReplayEvent[], timestamp: number): number {
  if (events.length === 0) return -1;
  let lo = 0;
  let hi = events.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (events[mid].timestamp <= timestamp) lo = mid;
    else hi = mid - 1;
  }
  return events[lo].timestamp <= timestamp ? lo : -1;
}

/** Filter coordination data to only events at or before playbackTime. */
export function filterAtTime<T>(
  items: T[],
  timestampKey: keyof T & string,
  playbackTime: number | null,
): T[] {
  if (playbackTime == null) return items;
  return items.filter((item) => {
    const ts = item[timestampKey] as unknown;
    if (typeof ts === "string") return new Date(ts).getTime() <= playbackTime;
    return true;
  });
}

import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories, coordSessions, coordFileClaims, scheduleRuns } from "../../../lib/db/schema";
import { gte, desc, eq, and } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

type RawEvent = {
  id: string;
  source: string;
  eventType: string;
  title: string;
  detail: string | null;
  timestamp: string;
  agentId: string | null;
  project: string | null;
  linkedTab?: string;
  linkedId?: string;
};

/**
 * GET /api/admin/timeline?hours=24&source=all&limit=50&offset=0
 *
 * Unified timeline merging events from coordination, memory, and jobs.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const hours = Math.min(168, Math.max(1, parseInt(url.searchParams.get("hours") ?? "24", 10)));
  const source = url.searchParams.get("source") ?? "all";
  const limit = Math.min(100, Math.max(10, parseInt(url.searchParams.get("limit") ?? "50", 10)));
  const offset = Math.max(0, parseInt(url.searchParams.get("offset") ?? "0", 10));

  const cutoff = new Date(Date.now() - hours * 60 * 60_000).toISOString();

  const allEvents: RawEvent[] = [];

  try {
    // 1. Coordination events (sessions)
    if (source === "all" || source === "coordination") {
      const sessions = await db
        .select()
        .from(coordSessions)
        .where(gte(coordSessions.startedAt, cutoff))
        .orderBy(desc(coordSessions.startedAt))
        .limit(50);

      for (const s of sessions) {
        allEvents.push({
          id: `coord-session-${s.id}`,
          source: "coordination",
          eventType: s.status === "ended" ? "session_ended" : "session_started",
          title: `Agent ${s.status === "ended" ? "ended" : "started"}: ${s.id.slice(0, 8)}`,
          detail: null,
          timestamp: s.startedAt || "",
          agentId: s.id,
          project: null,
          linkedTab: "coordination",
        });
      }
    }

    // 2. Memory events
    if (source === "all" || source === "memory") {
      const mems = await db
        .select({
          nodeId: memories.nodeId,
          content: memories.content,
          memoryType: memories.memoryType,
          metadata: memories.metadata,
          createdAt: memories.createdAt,
        })
        .from(memories)
        .where(gte(memories.createdAt, cutoff))
        .orderBy(desc(memories.createdAt))
        .limit(30);

      for (const m of mems) {
        let meta: Record<string, unknown> = {};
        try { meta = m.metadata ? JSON.parse(m.metadata) : {}; } catch { /* ignore */ }

        allEvents.push({
          id: `memory-${m.nodeId}`,
          source: "memory",
          eventType: `memory_${m.memoryType ?? "stored"}`,
          title: `Memory stored: ${m.memoryType ?? "unknown"}`,
          detail: (m.content ?? "").slice(0, 100),
          timestamp: m.createdAt,
          agentId: (meta.session_id as string) ?? null,
          project: (meta.project as string) ?? null,
          linkedTab: "feed",
          linkedId: m.nodeId,
        });
      }
    }

    // 3. Job events
    if (source === "all" || source === "job") {
      const runs = await db
        .select()
        .from(scheduleRuns)
        .where(gte(scheduleRuns.startedAt, cutoff))
        .orderBy(desc(scheduleRuns.startedAt))
        .limit(30);

      for (const r of runs) {
        allEvents.push({
          id: `job-${r.id}`,
          source: "job",
          eventType: r.status === "error" ? "job_failed" : r.status === "ok" ? "job_completed" : "job_started",
          title: `Job ${r.status}: ${r.scheduleId ?? "unknown"}`,
          detail: r.error ?? null,
          timestamp: r.finishedAt ?? r.startedAt ?? "",
          agentId: null,
          project: null,
          linkedTab: "jobs",
        });
      }
    }

    // Sort by timestamp descending
    allEvents.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    const total = allEvents.length;
    const paged = allEvents.slice(offset, offset + limit);

    return new Response(
      JSON.stringify({ events: paged, total, hasMore: offset + limit < total }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[timeline]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Internal error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

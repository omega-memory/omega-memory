import type { APIRoute } from "astro";
import type { InferSelectModel } from "drizzle-orm";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { coordSessions, memories } from "../../../../lib/db/schema";
import { desc, gte, eq, and, inArray, ne } from "drizzle-orm";

type CoordSessionRow = InferSelectModel<typeof coordSessions>;

/**
 * GET /api/admin/projects/timeline?project=omega&limit=20&offset=0&since=2026-02-01
 *
 * Returns a unified timeline of sessions for a given project.
 * The production route also queries coord_handoffs, coord_decisions, coord_git_events
 * which don't exist in the Drizzle schema. We use coord_sessions + memories fallback.
 */

const NOISE_TASKS = [
  "pre-edit guard claim",
  "pre-edit file guard",
  "file claim",
  "hook invocation",
];

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const project = url.searchParams.get("project");
  const limit = Math.min(parseInt(url.searchParams.get("limit") ?? "20", 10), 100);
  const offset = parseInt(url.searchParams.get("offset") ?? "0", 10);
  const since = url.searchParams.get("since");

  if (!project) {
    return new Response(
      JSON.stringify({ error: "project parameter required", entries: [], total: 0, project: "" }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  try {
    // Build conditions — match project via agentType as proxy
    const conditions = [eq(coordSessions.agentType, project)];
    if (since) {
      conditions.push(gte(coordSessions.startedAt, since));
    }

    // Fetch sessions (we can't easily exclude NOISE_TASKS with Drizzle without
    // a "task" column, so we filter in JS after fetch)
    const allSessions: CoordSessionRow[] = await db
      .select()
      .from(coordSessions)
      .where(and(...conditions))
      .orderBy(desc(coordSessions.startedAt))
      .limit(limit + offset + 50); // over-fetch for noise filtering

    // Filter out noise (metadata may contain task info)
    const filtered = allSessions.filter((s: CoordSessionRow) => {
      if (!s.metadata) return true;
      try {
        const meta = JSON.parse(s.metadata);
        return !NOISE_TASKS.includes(meta.task ?? "");
      } catch {
        return true;
      }
    });

    const total = filtered.length;
    const sessions = filtered.slice(offset, offset + limit);

    if (sessions.length === 0) {
      return new Response(
        JSON.stringify({ entries: [], total: 0, project }),
        {
          status: 200,
          headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
        },
      );
    }

    const sessionIds = sessions.map((s: CoordSessionRow) => s.id);

    // Try to enrich with memories fallback (handoff/session_summary/checkpoint)
    let memoryData: { sessionId: string | null; content: string; eventType: string | null; createdAt: string }[] = [];
    try {
      memoryData = await db
        .select({
          sessionId: memories.sessionId,
          content: memories.content,
          eventType: memories.eventType,
          createdAt: memories.createdAt,
        })
        .from(memories)
        .where(
          and(
            inArray(memories.sessionId, sessionIds),
            inArray(memories.eventType, ["handoff", "session_summary", "checkpoint"]),
          ),
        )
        .orderBy(desc(memories.createdAt))
        .limit(sessionIds.length * 3);
    } catch {
      // Best-effort fallback
    }

    // Build context map from memories
    const contextMap = new Map<string, string>();
    const priority: Record<string, number> = { handoff: 3, session_summary: 2, checkpoint: 1 };

    for (const sid of sessionIds) {
      const candidates = memoryData.filter((m) => m.sessionId === sid);
      if (candidates.length === 0) continue;
      candidates.sort(
        (a, b) => (priority[b.eventType ?? ""] ?? 0) - (priority[a.eventType ?? ""] ?? 0),
      );
      let ctx = (candidates[0].content ?? "")
        .replace(/^(Session handoff:|Session summary:|Auto-checkpoint:)\s*/i, "")
        .replace(/^#+\s+/gm, "")
        .trim();
      const firstLine = ctx.split("\n").find((l) => l.trim().length > 10);
      if (firstLine) {
        contextMap.set(sid, firstLine.trim().slice(0, 200));
      }
    }

    // Assemble timeline entries
    const entries = sessions.map((session: CoordSessionRow) => {
      const keyContext = contextMap.get(session.id) ?? null;
      const start = session.startedAt ? new Date(session.startedAt).getTime() : 0;
      const end = session.endedAt
        ? new Date(session.endedAt).getTime()
        : start;
      const durationMinutes = Math.round((end - start) / 60000);

      return {
        session: {
          session_id: session.id,
          started_at: session.startedAt,
          status: session.status,
          agent_type: session.agentType,
        },
        handoff: keyContext
          ? { key_context: keyContext, blocked_items: null, next_steps: null, git_branch: null }
          : null,
        decisions: [],
        gitEvents: [],
        durationMinutes,
      };
    });

    // Filter out empty sessions (no context and short duration)
    const meaningful = entries.filter((e) => {
      if (e.handoff) return true;
      if (e.durationMinutes >= 2) return true;
      return false;
    });

    return new Response(
      JSON.stringify({ entries: meaningful, total, project }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[projects/timeline]", msg);
    return new Response(
      JSON.stringify({ error: msg, entries: [], total: 0, project }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
};

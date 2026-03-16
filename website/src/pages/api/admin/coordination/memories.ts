import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories } from "../../../../lib/db/schema";
import { eq, desc, and } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/memories?session_id=xxx
 *
 * Returns recent memories created by a specific agent session.
 * Used by the coordination detail panel to show what an agent has learned/decided.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const params = new URL(context.request.url).searchParams;
  const sessionId = params.get("session_id");

  if (!sessionId) {
    return new Response(
      JSON.stringify({ error: "session_id query param required", memories: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  try {
    const rows = await db
      .select({
        id: memories.id,
        content: memories.content,
        eventType: memories.eventType,
        createdAt: memories.createdAt,
        project: memories.project,
        memoryType: memories.memoryType,
        priority: memories.priority,
        accessCount: memories.accessCount,
      })
      .from(memories)
      .where(and(eq(memories.sessionId, sessionId), eq(memories.userId, user!.id)))
      .orderBy(desc(memories.createdAt))
      .limit(20);

    return new Response(JSON.stringify({ memories: rows }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[coordination/memories GET]", message);
    return new Response(JSON.stringify({ error: message, memories: [] }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

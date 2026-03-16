import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { coordSessions, coordFileClaims } from "../../../../lib/db/schema";
import { inArray } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * DELETE /api/admin/coordination/sessions
 *
 * Deletes coordination sessions and their associated file claims.
 * Body: { session_ids: string[] }
 */
export const DELETE: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const sessionIds: string[] = body?.session_ids;

    if (!Array.isArray(sessionIds) || sessionIds.length === 0) {
      return new Response(JSON.stringify({ error: "session_ids array required" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    if (sessionIds.length > 100) {
      return new Response(JSON.stringify({ error: "Too many session_ids (max 100)" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    if (!sessionIds.every((id: string) => UUID_RE.test(id))) {
      return new Response(JSON.stringify({ error: "Invalid session_id format" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    // Delete file claims (no FK cascade)
    await db
      .delete(coordFileClaims)
      .where(inArray(coordFileClaims.sessionId, sessionIds));

    // coord_tasks not in schema — skip

    // Delete sessions
    await db
      .delete(coordSessions)
      .where(inArray(coordSessions.id, sessionIds));

    return new Response(JSON.stringify({ deleted: sessionIds.length }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[coordination/sessions DELETE]", message);
    return new Response(JSON.stringify({ error: message, deleted: 0 }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

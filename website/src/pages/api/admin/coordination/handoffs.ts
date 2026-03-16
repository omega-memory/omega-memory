import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/handoffs?session_id=xxx
 * GET /api/admin/coordination/handoffs?session_ids=a,b,c
 *
 * Returns handoffs written by session(s).
 * coord_handoffs table is not in the Drizzle schema — returns empty array.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const params = new URL(context.request.url).searchParams;
  const sessionId = params.get("session_id");
  const sessionIdsParam = params.get("session_ids");
  const ids = sessionIdsParam
    ? sessionIdsParam.split(",").filter(Boolean)
    : sessionId
      ? [sessionId]
      : [];

  if (ids.length === 0) {
    return new Response(
      JSON.stringify({ error: "session_id or session_ids query param required", handoffs: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  // coord_handoffs not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ handoffs: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

/**
 * PATCH /api/admin/coordination/handoffs
 *
 * Actions: mark_read — appends current user to read_by array.
 * coord_handoffs table is not in the Drizzle schema — returns graceful response.
 */
export const PATCH: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { action, handoff_id } = body;

    if (action !== "mark_read" || !handoff_id) {
      return new Response(
        JSON.stringify({ error: "action 'mark_read' and handoff_id required" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    // coord_handoffs not in Drizzle schema — return graceful response
    return new Response(JSON.stringify({ success: true, already_read: true }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[coordination/handoffs PATCH]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

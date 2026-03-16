import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/decisions?session_id=xxx
 * GET /api/admin/coordination/decisions?session_ids=a,b,c
 *
 * Returns decisions made by session(s).
 * coord_decisions table is not in the Drizzle schema — returns empty array.
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
      JSON.stringify({ error: "session_id or session_ids query param required", decisions: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  // coord_decisions not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ decisions: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

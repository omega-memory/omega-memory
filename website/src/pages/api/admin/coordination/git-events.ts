import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/git-events?session_id=xxx
 * GET /api/admin/coordination/git-events?session_ids=a,b,c
 *
 * Returns git events (commits, pushes, etc.) for session(s).
 * coord_git_events table is not in the Drizzle schema — returns empty array.
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
      JSON.stringify({ error: "session_id or session_ids query param required", git_events: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  // coord_git_events not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ git_events: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

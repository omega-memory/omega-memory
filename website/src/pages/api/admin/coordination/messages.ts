import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * GET /api/admin/coordination/messages?session_id=xxx
 * GET /api/admin/coordination/messages?session_ids=a,b,c
 *
 * Returns messages sent or received by session(s).
 * coord_messages table is not in the Drizzle schema — returns empty array.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const params = new URL(context.request.url).searchParams;
  const sessionId = params.get("session_id");
  const sessionIdsParam = params.get("session_ids");
  const ids = (sessionIdsParam
    ? sessionIdsParam.split(",").filter(Boolean)
    : sessionId
      ? [sessionId]
      : []
  ).filter((id) => UUID_RE.test(id));

  if (ids.length === 0) {
    return new Response(
      JSON.stringify({ error: "session_id or session_ids query param required", messages: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  // coord_messages not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ messages: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

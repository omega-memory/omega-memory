import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/audit?session_id=xxx
 *
 * Returns last 200 tool calls for a session, ordered by call_index ASC.
 * coord_audit table is not in the Drizzle schema — returns empty array.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const params = new URL(context.request.url).searchParams;
  const sessionId = params.get("session_id");

  if (!sessionId) {
    return new Response(
      JSON.stringify({ error: "session_id query param required", audit: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  // coord_audit not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ audit: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

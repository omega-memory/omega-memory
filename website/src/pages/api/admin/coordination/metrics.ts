import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/coordination/metrics?session_id=xxx
 * GET /api/admin/coordination/metrics?session_ids=a,b,c
 *
 * Returns coordination metrics for one or more sessions.
 * coord_metrics table is not in the Drizzle schema — returns empty array.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  // coord_metrics not in Drizzle schema — return empty gracefully
  return new Response(JSON.stringify({ metrics: [] }), {
    status: 200,
    headers: JSON_HEADERS,
  });
};

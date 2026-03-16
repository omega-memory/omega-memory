import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/audit-log
 *
 * Query params (two modes, mutually exclusive):
 *   label  - schedule_label, returns most recent events (desc)
 *   runId  - run UUID, returns chronological trail for one run (asc)
 *   limit  - max rows for label mode (default 50, max 200)
 *
 * The production route queries a "job_audit_log" table that doesn't exist in
 * the Drizzle schema. We return graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const label = url.searchParams.get("label");
  const runId = url.searchParams.get("runId");

  if (!label && !runId) {
    return new Response(
      JSON.stringify({ error: "label or runId parameter required" }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  try {
    // "job_audit_log" table not in Drizzle schema — return empty events
    return new Response(JSON.stringify({ events: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    console.error("[audit-log GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

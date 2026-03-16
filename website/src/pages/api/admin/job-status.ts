import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/job-status?id=<uuid>
 *
 * Returns the status and result of a pending_events job.
 * The production route queries a "pending_events" table that doesn't exist in
 * the Drizzle schema. We return a graceful 404 until the table is added.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const id = url.searchParams.get("id");

  if (!id) {
    return new Response(JSON.stringify({ error: "id parameter required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    // "pending_events" table not in Drizzle schema — return not found
    return new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[job-status]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

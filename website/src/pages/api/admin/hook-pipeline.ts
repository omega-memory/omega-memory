import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/hook-pipeline?days=7
 *
 * Returns hook execution DAG with stats. The production route queries an
 * "audit_log" table that doesn't exist in the Drizzle schema. We return
 * the static DAG structure with zero execution counts.
 */

// Hook execution order (DAG)
const HOOK_ORDER = [
  "session_start",
  "pre_edit",
  "post_edit",
  "pre_push",
  "post_push",
  "pre_file_guard",
  "session_stop",
];

const HOOK_EDGES: [string, string][] = [
  ["session_start", "pre_edit"],
  ["pre_edit", "post_edit"],
  ["post_edit", "pre_push"],
  ["pre_push", "post_push"],
  ["pre_edit", "pre_file_guard"],
];

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const days = Math.min(90, Math.max(1, parseInt(url.searchParams.get("days") ?? "7", 10)));

  try {
    // "audit_log" table not in Drizzle schema — return DAG structure with empty stats
    const nodes = HOOK_ORDER.map((id) => ({
      id,
      label: id
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase()),
      status: "inactive" as const,
      executionCount: 0,
      avgDurationMs: 0,
      lastExecuted: null,
      recentExecutions: [],
    }));

    const edges = HOOK_EDGES.map(([from, to]) => ({ from, to }));

    return new Response(
      JSON.stringify({ nodes, edges, periodDays: days }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[hook-pipeline]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Internal error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

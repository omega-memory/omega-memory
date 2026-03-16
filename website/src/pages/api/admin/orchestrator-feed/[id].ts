import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

/**
 * PATCH /api/admin/orchestrator-feed/[id] - approve, reject, or mark read
 *
 * The orchestrator feed depends on external tables not present in
 * the self-hosted Drizzle schema. Returns graceful 501.
 */
export const PATCH: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const id = context.params.id;
    const body = await context.request.json();
    const { action } = body;

    if (!action) {
      return new Response(JSON.stringify({ error: "action required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!["approve", "reject", "read"].includes(action)) {
      return new Response(
        JSON.stringify({ error: "Unknown action. Use 'approve', 'reject', or 'read'" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    if (action === "approve") {
      const { event_params } = body;
      if (!event_params || typeof event_params !== "object") {
        return new Response(JSON.stringify({ error: "event_params required for approve" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (!event_params.event_type || typeof event_params.event_type !== "string" || !event_params.event_type.trim()) {
        return new Response(
          JSON.stringify({ error: "event_params.event_type is required and must be a non-empty string" }),
          { status: 400, headers: { "Content-Type": "application/json" } },
        );
      }
    }

    return new Response(
      JSON.stringify({
        error: "Orchestrator feed not configured",
        message:
          "The orchestrator feed requires the orchestrator_feed_items table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[orchestrator-feed PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

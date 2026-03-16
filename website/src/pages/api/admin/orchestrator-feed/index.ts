import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

/**
 * GET /api/admin/orchestrator-feed?item_type=&status=&limit=&cursor=
 *
 * The orchestrator feed depends on external tables (orchestrator_feed_items)
 * not present in the self-hosted Drizzle schema. Returns graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // orchestrator_feed_items table not in self-hosted schema — return empty
    return new Response(
      JSON.stringify({
        items: [],
        hasMore: false,
        nextCursor: null,
        pendingProposals: 0,
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    console.error("[orchestrator-feed GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/**
 * POST /api/admin/orchestrator-feed - create a feed item
 *
 * Stub: orchestrator feed table not available in self-hosted schema.
 */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { item_type, title } = body;

    if (!item_type || !title) {
      return new Response(JSON.stringify({ error: "item_type and title required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const ALLOWED_ITEM_TYPES = ["update", "insight", "report", "alert", "proposal", "plan"];
    if (!ALLOWED_ITEM_TYPES.includes(item_type)) {
      return new Response(
        JSON.stringify({
          error: `Invalid item_type. Must be one of: ${ALLOWED_ITEM_TYPES.join(", ")}`,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
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
    console.error("[orchestrator-feed POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

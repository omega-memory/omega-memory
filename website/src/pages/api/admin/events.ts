import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/events?account=&status=&type=&limit=
 *
 * Events depend on the pending_events table which is not present in the
 * self-hosted Drizzle schema. Returns graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // pending_events table not in self-hosted schema — return empty
    return new Response(JSON.stringify({ events: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[events GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** POST /api/admin/events - create a new event */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { event_type, scheduled_for } = body;

    if (!event_type || !scheduled_for) {
      return new Response(
        JSON.stringify({ error: "event_type and scheduled_for required" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    if (isNaN(new Date(scheduled_for).getTime())) {
      return new Response(
        JSON.stringify({ error: "scheduled_for must be a valid ISO 8601 date string" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        error: "Events not configured",
        message:
          "The events feature requires the pending_events table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[events POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** PATCH /api/admin/events - cancel an event or trigger "run now" */
export const PATCH: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id, action } = body;

    if (!id) {
      return new Response(JSON.stringify({ error: "Event id required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!action || !["cancel", "run_now"].includes(action)) {
      return new Response(
        JSON.stringify({ error: "Unknown action. Use 'cancel' or 'run_now'" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        error: "Events not configured",
        message:
          "The events feature requires the pending_events table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[events PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

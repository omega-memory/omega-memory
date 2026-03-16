import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/autonomous-mode
 *
 * Autonomous mode depends on external tables (engagement_quota_config, tweets,
 * engagement_suggestions) not present in the self-hosted Drizzle schema.
 * Returns graceful default data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // engagement_quota_config table not in self-hosted schema — return defaults
    return new Response(
      JSON.stringify({
        enabled: false,
        tweets_today: 0,
        replies_today: 0,
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    console.error("[autonomous-mode GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/**
 * PATCH /api/admin/autonomous-mode
 * Toggle autonomous mode.
 *
 * Stub: engagement_quota_config table not available in self-hosted schema.
 */
export const PATCH: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    if (typeof body.enabled !== "boolean") {
      return new Response(JSON.stringify({ error: "enabled must be a boolean" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        error: "Autonomous mode not configured",
        message:
          "The autonomous mode feature requires the engagement_quota_config table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[autonomous-mode PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

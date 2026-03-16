import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/campaigns?account=
 *
 * Campaigns depend on external tables (campaigns, engagement_quota_config)
 * not present in the self-hosted Drizzle schema. Returns graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const account = new URL(context.request.url).searchParams.get("account");

    if (account) {
      return new Response(JSON.stringify({ campaign: null }), {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      });
    }

    return new Response(JSON.stringify({ campaigns: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[campaigns GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** POST /api/admin/campaigns - create a new campaign */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const account = body.account;
    if (!account) {
      return new Response(JSON.stringify({ error: "Account required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        error: "Campaigns not configured",
        message:
          "The campaigns feature requires the campaigns table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[campaigns POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** PATCH /api/admin/campaigns - update campaign fields */
export const PATCH: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id } = body;

    if (!id) {
      return new Response(JSON.stringify({ error: "Campaign id required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        error: "Campaigns not configured",
        message:
          "The campaigns feature requires the campaigns table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[campaigns PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

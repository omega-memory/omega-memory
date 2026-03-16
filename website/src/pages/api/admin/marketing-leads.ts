import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/marketing-leads?source=&format=&limit=&page=
 *
 * Marketing leads depend on the marketing_leads table which is not present
 * in the self-hosted Drizzle schema. Returns graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const params = new URL(context.request.url).searchParams;
    const format = params.get("format");
    const limit = Math.min(parseInt(params.get("limit") || "500", 10), 5000);
    const page = Math.max(parseInt(params.get("page") || "1", 10), 1);

    // CSV export (empty)
    if (format === "csv") {
      const header = "email,source,tags,created_at\n";
      return new Response(header, {
        headers: {
          "Content-Type": "text/csv",
          "Content-Disposition": `attachment; filename="marketing-leads-${new Date().toISOString().slice(0, 10)}.csv"`,
        },
      });
    }

    // marketing_leads table not in self-hosted schema — return empty
    return new Response(
      JSON.stringify({
        leads: [],
        total: 0,
        page,
        limit,
        breakdown: {},
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    console.error("[marketing-leads GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** POST /api/admin/marketing-leads - add tags or manually insert a lead */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();

    // Validate tag operation
    if (body.action === "tag" && Array.isArray(body.ids) && body.tag) {
      const tag = String(body.tag).trim().toLowerCase();
      if (!tag || tag.length > 50) {
        return new Response(JSON.stringify({ error: "Invalid tag" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    // Validate email insert
    if (body.email && body.source) {
      const email = String(body.email).trim().toLowerCase();
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        return new Response(JSON.stringify({ error: "Invalid email" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    return new Response(
      JSON.stringify({
        error: "Marketing leads not configured",
        message:
          "The marketing leads feature requires the marketing_leads table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[marketing-leads POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

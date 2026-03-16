import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

/**
 * GET/PATCH/POST/DELETE /api/admin/projects/registry
 *
 * The production route manages a "projects" table that doesn't exist in the
 * Drizzle schema. We return graceful empty data for GET and stub errors for
 * mutation methods until the projects table is added to the schema.
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // "projects" table not in Drizzle schema — return empty list
    return new Response(JSON.stringify({ projects: [] }), {
      status: 200,
      headers: { ...JSON_HEADERS, "Cache-Control": "private, no-store" },
    });
  } catch (err: unknown) {
    console.error("[projects/registry GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const PATCH: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id } = body;

    if (!id || typeof id !== "string") {
      return new Response(JSON.stringify({ error: "id is required" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    // "projects" table not in Drizzle schema
    return new Response(
      JSON.stringify({ error: "Projects table not available — registry is read-only in this deployment" }),
      { status: 501, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[projects/registry PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id, name } = body;

    if (!id || !name) {
      return new Response(JSON.stringify({ error: "id and name are required" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    // "projects" table not in Drizzle schema
    return new Response(
      JSON.stringify({ error: "Projects table not available — registry is read-only in this deployment" }),
      { status: 501, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[projects/registry POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const DELETE: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const url = new URL(context.request.url);
    const id = url.searchParams.get("id");

    if (!id) {
      return new Response(JSON.stringify({ error: "id parameter required" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    // "projects" table not in Drizzle schema
    return new Response(
      JSON.stringify({ error: "Projects table not available — registry is read-only in this deployment" }),
      { status: 501, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[projects/registry DELETE]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

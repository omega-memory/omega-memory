import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../lib/api/requireAuth";
import { schedules } from "../../lib/db/schema";
import { eq } from "drizzle-orm";

/**
 * GET /api/schedules — List all schedules.
 * Used by the sidebar badge polling for schedule counts.
 */
export const GET: APIRoute = async (context) => {
  const { db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const rows = await db.select().from(schedules).orderBy(schedules.name);

    return new Response(
      JSON.stringify({ schedules: rows }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[schedules GET]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: "Failed to fetch schedules", detail: err instanceof Error ? err.message : String(err) }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

/**
 * PATCH /api/schedules — Update a schedule.
 */
export const PATCH: APIRoute = async (context) => {
  const { db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id, ...fields } = body;

    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Build update payload from allowed fields
    const allowed = ["name", "cronExpr", "handler", "enabled", "metadata"];
    const update: Record<string, unknown> = {};
    for (const key of allowed) {
      if (key in fields) update[key] = fields[key];
    }

    if (Object.keys(update).length === 0) {
      return new Response(JSON.stringify({ error: "No valid fields to update" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await db.update(schedules).set(update).where(eq(schedules.id, id));

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    console.error("[schedules PATCH]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: "Failed to update schedule", detail: err instanceof Error ? err.message : String(err) }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

/**
 * DELETE /api/schedules — Soft-delete: disable the schedule.
 */
export const DELETE: APIRoute = async (context) => {
  const { db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { id } = body;

    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await db.update(schedules).set({ enabled: 0 }).where(eq(schedules.id, id));

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    console.error("[schedules DELETE]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: "Failed to delete schedule", detail: err instanceof Error ? err.message : String(err) }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

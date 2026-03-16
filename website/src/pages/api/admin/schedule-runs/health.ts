import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { schedules, scheduleRuns } from "../../../../lib/db/schema";
import { eq, ne, desc } from "drizzle-orm";

/**
 * GET /api/admin/schedule-runs/health
 *
 * Returns last 7 run statuses per schedule for health dot display.
 * Response: { health: Record<string, ("ok" | "error" | "completed" | "failed")[]> }
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // Get all schedules
    const allSchedules = await db.select().from(schedules);

    const health: Record<string, string[]> = {};

    for (const sched of allSchedules) {
      const runs = await db
        .select({
          status: scheduleRuns.status,
          startedAt: scheduleRuns.startedAt,
        })
        .from(scheduleRuns)
        .where(eq(scheduleRuns.scheduleId, sched.id))
        .orderBy(desc(scheduleRuns.startedAt))
        .limit(7);

      // Filter out "running" status and reverse for chronological order
      const statuses = runs
        .filter((r: { status: string | null; startedAt: string | null }) => r.status !== "running")
        .map((r: { status: string | null; startedAt: string | null }) => r.status ?? "unknown")
        .reverse();

      if (statuses.length > 0) {
        health[sched.name] = statuses;
      }
    }

    return new Response(JSON.stringify({ health }), {
      status: 200,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[schedule-runs/health]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

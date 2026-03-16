import type { APIRoute } from "astro";
import type { InferSelectModel } from "drizzle-orm";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { scheduleRuns, schedules } from "../../../../lib/db/schema";
import { eq, desc, gte, ne, and } from "drizzle-orm";

type ScheduleRunRow = InferSelectModel<typeof scheduleRuns>;

/**
 * GET /api/admin/schedule-runs
 *
 * Query params:
 *   label  - schedule name (required)
 *   limit  - max rows (default 20, max 100)
 *   days   - lookback period for summary mode (default 30)
 *   mode   - "list" (default) | "summary"
 *
 * POST /api/admin/schedule-runs
 *   Body: { label: string, action: "trigger" }
 *   Manual trigger — inserts a schedule_runs row with status "running".
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const label = url.searchParams.get("label");

  if (!label) {
    return new Response(JSON.stringify({ error: "label required" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  try {
    const mode = url.searchParams.get("mode") || "list";

    // Find the schedule by name
    const [schedule] = await db
      .select()
      .from(schedules)
      .where(eq(schedules.name, label))
      .limit(1);

    if (!schedule) {
      // No matching schedule — return empty results gracefully
      if (mode === "summary") {
        return new Response(
          JSON.stringify({
            summary: {
              totalRuns: 0, okCount: 0, errorCount: 0,
              totalCostUsd: 0, avgCostUsd: 0,
              totalInputTokens: 0, totalOutputTokens: 0,
              avgDurationMs: 0, periodDays: 30,
            },
            costSeries: [],
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }
      return new Response(JSON.stringify({ runs: [] }), {
        status: 200,
        headers: JSON_HEADERS,
      });
    }

    if (mode === "summary") {
      const days = Math.min(90, Math.max(1, parseInt(url.searchParams.get("days") || "30", 10) || 30));
      const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

      const runs = await db
        .select()
        .from(scheduleRuns)
        .where(
          and(
            eq(scheduleRuns.scheduleId, schedule.id),
            gte(scheduleRuns.startedAt, cutoff),
            ne(scheduleRuns.status, "running"),
          ),
        )
        .orderBy(desc(scheduleRuns.startedAt))
        .limit(5000);

      let totalDuration = 0;
      let okCount = 0;
      let errorCount = 0;

      for (const r of runs) {
        if (r.startedAt && r.finishedAt) {
          totalDuration +=
            new Date(r.finishedAt).getTime() - new Date(r.startedAt).getTime();
        }
        if (r.status === "ok" || r.status === "completed") okCount++;
        if (r.status === "error" || r.status === "failed") errorCount++;
      }

      return new Response(
        JSON.stringify({
          summary: {
            totalRuns: runs.length,
            okCount,
            errorCount,
            totalCostUsd: 0,
            avgCostUsd: 0,
            totalInputTokens: 0,
            totalOutputTokens: 0,
            avgDurationMs: runs.length > 0 ? Math.round(totalDuration / runs.length) : 0,
            periodDays: days,
          },
          costSeries: runs.map((r: ScheduleRunRow) => ({
            date: r.startedAt,
            cost: 0,
            status: r.status,
          })),
        }),
        { status: 200, headers: JSON_HEADERS },
      );
    }

    // Default: list mode
    const limit = Math.min(100, Math.max(1, parseInt(url.searchParams.get("limit") || "20", 10) || 20));

    const runs = await db
      .select()
      .from(scheduleRuns)
      .where(eq(scheduleRuns.scheduleId, schedule.id))
      .orderBy(desc(scheduleRuns.startedAt))
      .limit(limit);

    return new Response(JSON.stringify({ runs }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[schedule-runs GET]", err instanceof Error ? err.message : err);
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
    const { label, action } = body;

    if (!label || action !== "trigger") {
      return new Response(
        JSON.stringify({ error: "label and action='trigger' required" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    // Find the schedule
    const [schedule] = await db
      .select()
      .from(schedules)
      .where(eq(schedules.name, label))
      .limit(1);

    if (!schedule) {
      return new Response(
        JSON.stringify({ error: "Schedule not found" }),
        { status: 404, headers: JSON_HEADERS },
      );
    }

    // Insert a new run with status "running"
    const runId = crypto.randomUUID();
    await db.insert(scheduleRuns).values({
      id: runId,
      scheduleId: schedule.id,
      status: "running",
      startedAt: new Date().toISOString(),
      metadata: JSON.stringify({ source: "manual" }),
    });

    return new Response(JSON.stringify({ success: true, runId }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[schedule-runs POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

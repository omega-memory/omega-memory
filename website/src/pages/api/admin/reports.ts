import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories } from "../../../lib/db/schema";
import { desc, eq } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json" };

/**
 * GET /api/admin/reports?limit=10
 *
 * Returns weekly report memories. In the production environment this
 * delegates to a dedicated `listWeeklyReports` helper backed by a
 * `weekly_reports` Supabase table. Since that table is not in the
 * self-hosted Drizzle schema, we surface memories with event_type
 * "research_report" or "session_summary" as a reasonable proxy.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const limit = parseInt(url.searchParams.get("limit") || "10", 10);

  try {
    // Fetch report-like memories as proxy for weekly_reports table
    const reports = await db
      .select({
        id: memories.id,
        nodeId: memories.nodeId,
        content: memories.content,
        eventType: memories.eventType,
        createdAt: memories.createdAt,
        project: memories.project,
      })
      .from(memories)
      .where(eq(memories.eventType, "research_report"))
      .orderBy(desc(memories.createdAt))
      .limit(limit);

    return new Response(JSON.stringify({ reports }), {
      status: 200,
      headers: { ...JSON_HEADERS, "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[reports GET]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

/**
 * POST /api/admin/reports - trigger report generation
 *
 * In production this inserts into `pending_events` (Supabase).
 * Since that table is not in the self-hosted schema, return a
 * graceful "not available" response.
 */
export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  return new Response(
    JSON.stringify({
      error: "Report generation not available in self-hosted mode",
      message: "The pending_events queue is only available in the managed cloud environment.",
    }),
    { status: 501, headers: JSON_HEADERS },
  );
};

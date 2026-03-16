import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { adminAlerts } from "../../../lib/db/schema";
import { gte, desc, eq } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/alert-history?status=all
 *
 * Returns admin alerts from the last 7 days with recurrence counts.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const status = url.searchParams.get("status") ?? "all";

  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60_000).toISOString();

  try {
    let rows;
    if (status !== "all") {
      rows = await db
        .select({
          id: adminAlerts.id,
          type: adminAlerts.type,
          severity: adminAlerts.severity,
          message: adminAlerts.message,
          status: adminAlerts.status,
          createdAt: adminAlerts.createdAt,
          resolvedAt: adminAlerts.resolvedAt,
        })
        .from(adminAlerts)
        .where(eq(adminAlerts.status, status))
        .orderBy(desc(adminAlerts.createdAt))
        .limit(50);
    } else {
      rows = await db
        .select({
          id: adminAlerts.id,
          type: adminAlerts.type,
          severity: adminAlerts.severity,
          message: adminAlerts.message,
          status: adminAlerts.status,
          createdAt: adminAlerts.createdAt,
          resolvedAt: adminAlerts.resolvedAt,
        })
        .from(adminAlerts)
        .orderBy(desc(adminAlerts.createdAt))
        .limit(50);
    }

    // Filter to last 7 days in JS (simpler than composing WHERE conditions)
    const alerts = rows.filter((a: typeof rows[number]) => a.createdAt && a.createdAt >= weekAgo);

    // Calculate recurrence counts by type
    const recurrenceCounts: Record<string, number> = {};
    for (const a of alerts) {
      recurrenceCounts[a.type] = (recurrenceCounts[a.type] ?? 0) + 1;
    }

    return new Response(JSON.stringify({ alerts, recurrenceCounts }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[alert-history]", err);
    return new Response(JSON.stringify({ error: "Internal error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

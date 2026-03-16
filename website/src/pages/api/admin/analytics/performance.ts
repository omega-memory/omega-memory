import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

/**
 * GET /api/admin/analytics/performance?account=&start=&end=
 *
 * Performance analytics depend on external metrics tables not present
 * in the self-hosted Drizzle schema. Returns graceful empty data
 * matching the PerformanceInsights shape.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // Return empty PerformanceInsights shape
    return new Response(
      JSON.stringify({
        byContentType: [],
        byTheme: [],
        bySlot: [],
        topPerformers: [],
        overall: {
          totalTracked: 0,
          avgEngagementRate: 0,
          recentTrend: "steady",
        },
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    console.error("[analytics/performance GET]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({
        byContentType: [],
        byTheme: [],
        bySlot: [],
        topPerformers: [],
        overall: { totalTracked: 0, avgEngagementRate: 0, recentTrend: "steady" },
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  }
};

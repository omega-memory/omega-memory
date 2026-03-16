import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";

/**
 * POST /api/admin/research/scan -- trigger research scanner
 *
 * The research scanner (runResearchScan) is a production-only dependency.
 * This stub returns a 501 directing callers to configure the scanner service.
 */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // runResearchScan is not available in the Astro self-hosted dashboard.
    // Return a helpful message instead of a hard crash.
    return new Response(
      JSON.stringify({
        error: "Research scanner not configured",
        message:
          "The research scanner requires external service configuration. " +
          "Set up the RESEARCH_SCANNER_URL environment variable to enable this endpoint.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[research/scan POST]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : "Scan failed" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

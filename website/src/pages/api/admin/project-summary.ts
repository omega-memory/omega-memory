import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * POST /api/admin/project-summary
 *
 * The production route uses llmComplete() to generate AI summaries and caches
 * them in a "project_summaries" table. Neither the LLM helper nor the cache
 * table exist in this deployment, so we return null gracefully.
 *
 * Body: { project: string, memories?: { content: string; type: string }[] }
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    let body: { project?: string; memories?: { content: string; type: string }[] };
    try {
      body = await context.request.json();
    } catch {
      return new Response(JSON.stringify({ error: "Invalid JSON" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    const project = body.project;
    if (!project) {
      return new Response(JSON.stringify({ error: "project is required" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    // LLM summarization and project_summaries cache not available in this deployment
    return new Response(JSON.stringify({ summary: null }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[project-summary]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ summary: null }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  }
};

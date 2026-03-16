import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories } from "../../../lib/db/schema";
import { eq, and, sql } from "drizzle-orm";

/**
 * POST /api/admin/ingest-url - ingest a URL as a research finding
 *
 * In production, this uses Grok for content extraction and scoreUrlIngest
 * for relevance scoring. The self-hosted version stores a placeholder
 * finding and returns without scoring (those services require API keys).
 */
export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const { url, project } = body as { url?: string; project?: string };

    if (!url || typeof url !== "string") {
      return new Response(JSON.stringify({ error: "URL is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Validate URL format
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      return new Response(JSON.stringify({ error: "Invalid URL format" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (!["http:", "https:"].includes(parsed.protocol)) {
      return new Response(JSON.stringify({ error: "URL must be http or https" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const domain = parsed.hostname.replace(/^www\./, "");

    // Dedup check: has this URL already been ingested?
    const existing = await db
      .select({ id: memories.id })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source_url') = ${url}`,
        ),
      )
      .limit(1);

    if (existing.length > 0) {
      return new Response(
        JSON.stringify({ error: "This URL has already been ingested" }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      );
    }

    // Without Grok/LLM, store the URL as a finding stub
    const content = `URL ingested: ${url}\n\nDomain: ${domain}\n\nContent extraction requires LLM provider configuration (GROK_API_KEY).`;

    const nodeId = `url_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    const result = await db
      .insert(memories)
      .values({
        nodeId,
        content,
        eventType: "research_report",
        priority: 3,
        project: project || null,
        userId: user!.id,
        createdAt: new Date().toISOString(),
        metadata: JSON.stringify({
          tag: "url_ingest",
          source_type: "url_ingest",
          source_url: url,
          source_domain: domain,
          ingested_at: new Date().toISOString(),
        }),
      })
      .returning({ id: memories.id });

    return new Response(
      JSON.stringify({
        ok: true,
        id: result[0]?.id,
        domain,
        scores: null,
        message: "URL recorded. Content extraction requires LLM provider configuration.",
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      },
    );
  } catch (err: unknown) {
    console.error("[ingest-url POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "URL ingestion failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

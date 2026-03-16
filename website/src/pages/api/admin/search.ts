import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories, documents } from "../../../lib/db/schema";
import { desc, like, and, or } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json" };

/**
 * POST /api/admin/search - Memory and document search (LIKE-based fallback)
 *
 * Body: { query: string, limit?: number, type?: "all" | "memories" | "documents" }
 *
 * In production this uses PostgreSQL tsvector full-text search.
 * For SQLite/D1, we fall back to LIKE-based content search.
 */
export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  let body: { query?: string; limit?: number; type?: string };
  try {
    body = await context.request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  const { query, limit: rawLimit = 10, type = "all" } = body;

  if (!query) {
    return new Response(JSON.stringify({ error: "query is required" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  const limit = Math.min(Math.max(1, rawLimit), 100);

  try {
    const results: { memories: Record<string, unknown>[]; documents: Record<string, unknown>[] } = {
      memories: [],
      documents: [],
    };

    // Search memories using LIKE (SQLite-compatible fallback for tsvector)
    if (type === "all" || type === "memories") {
      const words = query.split(/\s+/).filter((w: string) => w.length > 1);
      if (words.length > 0) {
        // Build LIKE conditions for each word
        const conditions = words.map((w) => like(memories.content, `%${w}%`));

        const rows = await db
          .select({
            id: memories.nodeId,
            content: memories.content,
            eventType: memories.eventType,
            priority: memories.priority,
            createdAt: memories.createdAt,
          })
          .from(memories)
          .where(and(...conditions))
          .orderBy(desc(memories.createdAt))
          .limit(limit);

        results.memories = rows.map((m) => ({
          memory_id: m.id,
          content: m.content,
          event_type: m.eventType,
          priority: m.priority,
          similarity: null,
        }));
      }
    }

    // Search documents by title/sourcePath
    if (type === "all" || type === "documents") {
      const pattern = `%${query}%`;
      const titleRows = await db
        .select({
          id: documents.id,
          title: documents.title,
          sourcePath: documents.sourcePath,
          sourceType: documents.sourceType,
          createdAt: documents.createdAt,
        })
        .from(documents)
        .where(or(like(documents.title, pattern), like(documents.sourcePath, pattern)))
        .orderBy(desc(documents.createdAt))
        .limit(limit);

      const seen = new Set<string>();
      for (const d of titleRows) {
        if (seen.has(d.id)) continue;
        seen.add(d.id);
        results.documents.push({
          id: d.id,
          content: d.title || d.sourcePath,
          document_title: d.title || d.sourcePath,
          source_path: d.sourcePath,
          mime_type: d.sourceType,
          similarity: null,
        });
      }
    }

    return new Response(JSON.stringify({ results, search_type: "like" }), {
      status: 200,
      headers: { ...JSON_HEADERS, "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[search POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

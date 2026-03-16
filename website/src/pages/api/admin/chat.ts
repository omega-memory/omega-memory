import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories } from "../../../lib/db/schema";
import { eq, desc, and, like, or } from "drizzle-orm";

/**
 * POST /api/admin/chat - RAG chat using memories as context
 *
 * Retrieves relevant memories via keyword search, then returns a
 * placeholder response. The full streaming LLM integration requires
 * configuring ANTHROPIC_API_KEY or equivalent.
 */
export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const { message, history: rawHistory = [] } = await context.request.json();

    if (!message) {
      return new Response(JSON.stringify({ error: "message is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Validate and filter history entries
    const history = (Array.isArray(rawHistory) ? rawHistory : []).filter(
      (entry: unknown): entry is { role: "user" | "assistant"; content: string } =>
        typeof entry === "object" &&
        entry !== null &&
        ((entry as Record<string, unknown>).role === "user" ||
          (entry as Record<string, unknown>).role === "assistant") &&
        typeof (entry as Record<string, unknown>).content === "string",
    );

    // Retrieve context via keyword search (scoped to this user)
    const words = message
      .split(/\s+/)
      .filter((w: string) => w.length > 2)
      .slice(0, 5);

    let contextLines: string[] = [];
    if (words.length > 0) {
      const conditions = words.map((w: string) => like(memories.content, `%${w}%`));
      const userCondition = user ? eq(memories.userId, user.id) : undefined;

      const rows = await db
        .select({
          content: memories.content,
          eventType: memories.eventType,
        })
        .from(memories)
        .where(userCondition ? and(userCondition, or(...conditions)) : or(...conditions))
        .orderBy(desc(memories.createdAt))
        .limit(10);

      contextLines = rows.map(
        (m: typeof rows[number]) => `- [${m.eventType || "memory"}] ${m.content}`,
      );
    }

    const memoryContext =
      contextLines.length > 0
        ? "## Relevant Memories\n\n" + contextLines.join("\n")
        : "No relevant context found.";

    // Without an LLM provider configured, return the retrieved context
    // so the frontend can still display something useful.
    // To enable full streaming, set ANTHROPIC_API_KEY and integrate llmStream.
    const encoder = new TextEncoder();
    const fallbackMessage =
      `I found ${contextLines.length} relevant memories for your query.\n\n` +
      (contextLines.length > 0
        ? memoryContext
        : "No matching memories found. Try a different query.") +
      "\n\n---\n*Note: Full AI chat requires LLM provider configuration (ANTHROPIC_API_KEY).*";

    const readable = new ReadableStream({
      start(controller) {
        const event = {
          type: "content_block_delta",
          delta: { type: "text_delta", text: fallbackMessage },
        };
        controller.enqueue(
          encoder.encode(
            `event: content_block_delta\ndata: ${JSON.stringify(event)}\n\n`,
          ),
        );
        controller.close();
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  } catch (err: unknown) {
    console.error("[chat POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Chat failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

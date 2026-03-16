import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories, entityIndex } from "../../../../lib/db/schema";
import { eq, isNotNull, and, inArray } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * GET /api/admin/coordination/entities?session_id=xxx
 *
 * Returns entities linked to a session's memories.
 * Derives the link through memories.sessionId + memories.entityId -> entityIndex.
 * Note: The production route uses a Supabase "entities" table; the Drizzle schema
 * has "entityIndex" instead. We use entityIndex for name/type lookups.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const params = new URL(context.request.url).searchParams;
  const sessionId = params.get("session_id");

  if (!sessionId) {
    return new Response(
      JSON.stringify({ error: "session_id query param required", entities: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  if (!UUID_RE.test(sessionId)) {
    return new Response(
      JSON.stringify({ error: "session_id must be a valid UUID", entities: [] }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  try {
    // Find distinct entity_ids from this session's memories
    const memData = await db
      .select({ entityId: memories.entityId })
      .from(memories)
      .where(and(eq(memories.sessionId, sessionId), isNotNull(memories.entityId)));

    // Deduplicate entity_ids and count
    const entityCounts = new Map<string, number>();
    for (const m of memData) {
      if (m.entityId) {
        entityCounts.set(m.entityId, (entityCounts.get(m.entityId) || 0) + 1);
      }
    }

    if (entityCounts.size === 0) {
      return new Response(JSON.stringify({ entities: [] }), {
        status: 200,
        headers: JSON_HEADERS,
      });
    }

    // entityIndex uses entityName as PK, but memories.entityId may reference by name
    const entityNames = [...entityCounts.keys()];
    const entData = await db
      .select({
        entityName: entityIndex.entityName,
        entityType: entityIndex.entityType,
      })
      .from(entityIndex)
      .where(inArray(entityIndex.entityName, entityNames));

    const entities = entData.map((e) => ({
      id: e.entityName,
      name: e.entityName,
      entityType: e.entityType,
      memoryCount: entityCounts.get(e.entityName) || 0,
    }));

    // Sort by memory count desc
    entities.sort((a, b) => b.memoryCount - a.memoryCount);

    return new Response(JSON.stringify({ entities }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[coordination/entities GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error", entities: [] }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

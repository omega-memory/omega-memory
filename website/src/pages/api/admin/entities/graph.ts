import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories, entityIndex, edges } from "../../../../lib/db/schema";
import { isNotNull, desc, or, eq } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/entities/graph
 *
 * Returns entity nodes and relationship links for force-directed graph visualization.
 * Uses entity_index table + edges table to build the graph.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // Fetch entities and memory links in parallel
    const [allEntities, memLinks] = await Promise.all([
      db.select().from(entityIndex),
      db
        .select({ entityId: memories.entityId })
        .from(memories)
        .where(isNotNull(memories.entityId)),
    ]);

    // Build set of entity names that appear in memories
    const userEntityNames = new Set<string>();
    for (const m of memLinks) {
      if (m.entityId) userEntityNames.add(m.entityId);
    }

    // Filter entities to those referenced by memories (if any links exist)
    const entities = userEntityNames.size > 0
      ? allEntities.filter((e: typeof allEntities[number]) => userEntityNames.has(e.entityName))
      : allEntities;

    const entityNameSet = new Set(entities.map((e: typeof entities[number]) => e.entityName));

    // Build nodes
    const nodes = entities.map((e: typeof entities[number]) => ({
      id: e.entityName,
      name: e.entityName,
      entityType: e.entityType || "unknown",
      status: null,
      metadata: e.metadata,
      createdAt: e.firstSeen,
      updatedAt: e.lastUpdated,
    }));

    // Fetch edges where source or target matches entity names
    // The edges table uses node IDs (memory node_ids), so we look for entity-related edges
    const allEdges = await db.select().from(edges).limit(5000);

    // Filter edges to those connecting known entities
    const links = allEdges
      .filter((e) => entityNameSet.has(e.sourceId) && entityNameSet.has(e.targetId))
      .map((e) => ({
        source: e.sourceId,
        target: e.targetId,
        type: e.edgeType,
        metadata: e.metadata,
      }));

    // Aggregate entity types for filter chips
    const typeCounts = new Map<string, number>();
    for (const e of entities) {
      const t = e.entityType || "unknown";
      typeCounts.set(t, (typeCounts.get(t) || 0) + 1);
    }
    const entityTypes = [...typeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({ type, count }));

    const truncated = userEntityNames.size > entities.length;

    return new Response(
      JSON.stringify({
        nodes,
        links,
        entityTypes,
        truncated,
        total: allEntities.length,
        returned: entities.length,
      }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[entities/graph GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

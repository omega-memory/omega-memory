import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories, entityIndex } from "../../../../lib/db/schema";
import { eq, isNotNull, desc, sql } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/entities?type=person&project=omega&page=1&limit=100
 *
 * Returns entities from the entity_index table enriched with:
 * - memory count per entity
 * - project associations (derived from linked memories)
 * - last seen (most recent memory created_at)
 *
 * Pagination: page (default 1), limit (default 200, max 2000)
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const typeFilter = url.searchParams.get("type");
  const projectFilter = url.searchParams.get("project");
  const page = Math.max(1, parseInt(url.searchParams.get("page") || "1", 10) || 1);
  const limit = Math.min(2000, Math.max(1, parseInt(url.searchParams.get("limit") || "200", 10) || 200));

  try {
    // Fetch all entities from entity_index
    const allEntities = await db.select().from(entityIndex);

    // Fetch memory links: memories with entity_id set (scoped to user)
    const memLinks = await db
      .select({
        entityId: memories.entityId,
        project: memories.project,
        createdAt: memories.createdAt,
      })
      .from(memories)
      .where(isNotNull(memories.entityId));

    // Build set of entity names that appear in user's memories
    const userEntityNames = new Set<string>();
    for (const m of memLinks) {
      if (m.entityId) userEntityNames.add(m.entityId);
    }

    // Build memory counts, project associations, and last seen per entity
    const memCounts = new Map<string, number>();
    const entityProjects = new Map<string, Set<string>>();
    const lastSeen = new Map<string, string>();

    for (const m of memLinks) {
      if (!m.entityId) continue;
      memCounts.set(m.entityId, (memCounts.get(m.entityId) || 0) + 1);

      if (m.project) {
        if (!entityProjects.has(m.entityId)) entityProjects.set(m.entityId, new Set());
        entityProjects.get(m.entityId)!.add(m.project);
      }

      const existing = lastSeen.get(m.entityId);
      if (!existing || (m.createdAt && m.createdAt > existing)) {
        lastSeen.set(m.entityId, m.createdAt);
      }
    }

    // Enrich entities
    let enriched = allEntities.map((e) => {
      const projects = entityProjects.get(e.entityName);
      const projectNames = projects
        ? [...projects].map((p) => {
            const trimmed = p.replace(/\/+$/, "").trim();
            if (!trimmed || trimmed === "/") return "Unknown";
            const last = trimmed.split("/").pop();
            return last || "Unknown";
          })
        : [];

      return {
        id: e.entityName,
        name: e.entityName,
        entityType: e.entityType || "unknown",
        status: null,
        metadata: e.metadata,
        createdAt: e.firstSeen,
        updatedAt: e.lastUpdated,
        memoryCount: memCounts.get(e.entityName) || 0,
        relationshipCount: 0,
        statementCount: e.statementCount || 0,
        outcomeCount: e.outcomeCount || 0,
        contradictionScore: e.contradictionScore || 0,
        followThroughRate: e.followThroughRate || 0,
        projects: projectNames,
        lastSeen: lastSeen.get(e.entityName) || null,
      };
    });

    // Apply filters
    if (typeFilter) {
      enriched = enriched.filter((e) => e.entityType === typeFilter);
    }
    if (projectFilter) {
      enriched = enriched.filter((e) =>
        e.projects.some((p) => p.toLowerCase() === projectFilter.toLowerCase()),
      );
    }

    // Sort by memory count desc, then name
    enriched.sort((a, b) => b.memoryCount - a.memoryCount || a.name.localeCompare(b.name));

    // Aggregate stats
    const typeCounts = new Map<string, number>();
    const projectCounts = new Map<string, number>();
    for (const e of enriched) {
      typeCounts.set(e.entityType, (typeCounts.get(e.entityType) || 0) + 1);
    }
    for (const e of enriched) {
      for (const p of e.projects) {
        projectCounts.set(p, (projectCounts.get(p) || 0) + 1);
      }
    }

    const entityTypes = [...typeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({ type, count }));

    const projectList = [...projectCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([project, count]) => ({ project, count }));

    // Paginate
    const totalFiltered = enriched.length;
    const offset = (page - 1) * limit;
    const paginated = enriched.slice(offset, offset + limit);

    return new Response(
      JSON.stringify({
        entities: paginated,
        total: allEntities.length,
        totalUserEntities: allEntities.length,
        filtered: totalFiltered,
        returned: paginated.length,
        truncated: totalFiltered > offset + limit,
        page,
        limit,
        entityTypes,
        projects: projectList,
      }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[entities GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

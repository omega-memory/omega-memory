import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories, edges, entityIndex } from "../../../../lib/db/schema";
import { desc, eq, inArray } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

const DEFAULT_MEMORY_LIMIT = 5000;
const MAX_MEMORY_LIMIT = 10000;
const MIN_MEMORIES = 5;

interface MemoryNode {
  id: string;
  content: string;
  event_type: string | null;
  project: string | null;
  priority: number;
  tags: string[] | null;
  created_at: string;
  session_id: string | null;
  updated_at: string | null;
  memory_type: string | null;
  access_count: number;
  access_rank: number;
  entity_id: string | null;
  is_superseded: boolean;
  confidence: number | null;
  observation: string | null;
  facts: string[] | null;
  feedback_score: number | null;
  category: string | null;
}

type EdgeType = "session" | "tag" | "related" | "contains_fact" | "same_entity" | "temporal_cluster" | "evolution" | "contradicts" | "supersedes" | "references_entity";

interface GraphLink {
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
}

function buildSynthEdges(nodes: MemoryNode[]): GraphLink[] {
  const links: GraphLink[] = [];
  const seen = new Set<string>();

  // 1. Session edges: link consecutive memories within same session
  const bySession = new Map<string, MemoryNode[]>();
  for (const node of nodes) {
    if (!node.session_id) continue;
    const group = bySession.get(node.session_id) || [];
    group.push(node);
    bySession.set(node.session_id, group);
  }

  for (const group of bySession.values()) {
    group.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
    for (let i = 0; i < group.length - 1; i++) {
      const key = `${group[i].id}-${group[i + 1].id}`;
      if (!seen.has(key)) {
        links.push({ source: group[i].id, target: group[i + 1].id, type: "session" });
        seen.add(key);
      }
    }
  }

  // 2. Tag edges: link memories sharing extracted_keywords (cap 5 links per keyword)
  const byTag = new Map<string, string[]>();
  for (const node of nodes) {
    if (!node.tags) continue;
    for (const tag of node.tags) {
      const group = byTag.get(tag) || [];
      group.push(node.id);
      byTag.set(tag, group);
    }
  }

  for (const ids of byTag.values()) {
    if (ids.length < 2) continue;
    let count = 0;
    for (let i = 0; i < ids.length && count < 5; i++) {
      for (let j = i + 1; j < ids.length && count < 5; j++) {
        const key = [ids[i], ids[j]].sort().join("-");
        if (!seen.has(key)) {
          links.push({ source: ids[i], target: ids[j], type: "tag" });
          seen.add(key);
          count++;
        }
      }
    }
  }

  return links;
}

/** Normalize a raw project path to a clean display label. */
function normalizeProject(raw: string): string | null {
  if (!raw) return null;
  let label = raw.trim();
  label = label.replace(/\/\.claude\/worktrees\/[^/]+\/?/g, "/");
  label = label.replace(/\/\.worktrees\/[^/]+\/?/g, "/");
  const pm = label.match(/\/Users\/[^/]+\/Projects\/(.+)/);
  if (pm) {
    label = pm[1];
  } else {
    const hm = label.match(/\/Users\/[^/]+\/(.+)/);
    if (hm) {
      label = hm[1];
    } else if (/^\/Users\/[^/]+\/?$/.test(label)) {
      return null;
    }
  }
  label = label.replace(/\/+/g, "/").replace(/^\/|\/$/g, "");
  if (!label || label === "/" || /^\d+$/.test(label)) return null;
  return label;
}

/**
 * GET /api/admin/memories/graph?project=<slug>&limit=5000
 *
 * Returns memory nodes, synthesized + real edges, and entity nodes
 * for force-directed graph visualization.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const url = new URL(context.request.url);
    const projectFilter = url.searchParams.get("project");
    const limit = Math.min(
      parseInt(url.searchParams.get("limit") || String(DEFAULT_MEMORY_LIMIT), 10),
      MAX_MEMORY_LIMIT,
    );

    // Build query for memories
    let allMemories;
    if (projectFilter) {
      const rawProjectValues = projectFilter.split(",").filter(Boolean);
      if (rawProjectValues.length === 1) {
        allMemories = await db
          .select()
          .from(memories)
          .where(eq(memories.project, rawProjectValues[0]))
          .orderBy(desc(memories.createdAt))
          .limit(limit);
      } else {
        allMemories = await db
          .select()
          .from(memories)
          .where(inArray(memories.project, rawProjectValues))
          .orderBy(desc(memories.createdAt))
          .limit(limit);
      }
    } else {
      allMemories = await db
        .select()
        .from(memories)
        .orderBy(desc(memories.createdAt))
        .limit(limit);
    }

    // Compute access_rank: normalize access_count to 0-1
    let maxAccess = 1;
    for (const m of allMemories) {
      const ac = m.accessCount ?? 0;
      if (ac > maxAccess) maxAccess = ac;
    }

    // Parse extracted_keywords JSON string into tags array
    function parseTags(raw: string | null): string[] | null {
      if (!raw) return null;
      try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : null;
      } catch {
        return null;
      }
    }

    function parseMeta(raw: string | null): Record<string, unknown> {
      if (!raw) return {};
      try {
        const parsed = JSON.parse(raw);
        return typeof parsed === "object" && parsed !== null ? parsed : {};
      } catch {
        return {};
      }
    }

    const nodes: MemoryNode[] = allMemories.map((m) => {
      const meta = parseMeta(m.metadata);
      const accessCount = m.accessCount ?? 0;
      const tags = parseTags(m.extractedKeywords);
      return {
        id: m.nodeId,
        content: m.content,
        event_type: m.eventType,
        project: m.project,
        priority: m.priority ?? 3,
        tags,
        created_at: m.createdAt,
        updated_at: null,
        session_id: m.sessionId,
        memory_type: m.memoryType ?? null,
        access_count: accessCount,
        access_rank: accessCount / maxAccess,
        entity_id: m.entityId ?? null,
        is_superseded: Boolean(meta.superseded || meta.superseded_by),
        confidence: typeof meta.capture_confidence === "number" ? meta.capture_confidence : null,
        observation: typeof meta.observation === "string" ? meta.observation : null,
        facts: Array.isArray(meta.facts) ? meta.facts as string[] : null,
        feedback_score: typeof meta.feedback_score === "number" ? meta.feedback_score : null,
        category: typeof meta.category === "string" ? meta.category : null,
      };
    });

    // Build synthesized edges
    const links = buildSynthEdges(nodes);

    // Fetch real edges from the edges table
    const nodeIds = new Set(nodes.map((n) => n.id));
    const allEdges = await db.select().from(edges).limit(20000);

    const seen = new Set(links.map((l) => `${l.source}-${l.target}-${l.type}`));
    for (const edge of allEdges) {
      if (!nodeIds.has(edge.sourceId) || !nodeIds.has(edge.targetId)) continue;
      const key = `${edge.sourceId}-${edge.targetId}-${edge.edgeType}`;
      if (seen.has(key)) continue;
      seen.add(key);
      links.push({
        source: edge.sourceId,
        target: edge.targetId,
        type: edge.edgeType as EdgeType,
        weight: edge.weight ?? 1.0,
      });
    }

    // Fetch entity nodes for memories that reference an entity_id
    const entityIds = [...new Set(nodes.map((n) => n.entity_id).filter((id): id is string => id !== null))];
    const entityNodes: { id: string; label: string; entity_type: string; group: string }[] = [];

    if (entityIds.length > 0) {
      const entityRows = await db
        .select()
        .from(entityIndex)
        .where(inArray(entityIndex.entityName, entityIds));

      for (const ent of entityRows) {
        entityNodes.push({
          id: ent.entityName,
          label: ent.entityName,
          entity_type: ent.entityType || "unknown",
          group: "entity",
        });
      }
      const entityNameSet = new Set(entityRows.map((e) => e.entityName));
      for (const node of nodes) {
        if (node.entity_id && entityNameSet.has(node.entity_id)) {
          links.push({
            source: node.id,
            target: node.entity_id,
            type: "references_entity",
          });
        }
      }
    }

    // Count memories per raw project value
    const rawCounts = new Map<string, number>();
    for (const n of nodes) {
      if (!n.project) continue;
      rawCounts.set(n.project, (rawCounts.get(n.project) || 0) + 1);
    }

    // Group by normalized display name
    const labelGroups = new Map<string, { rawValues: string[]; count: number }>();
    for (const [raw, count] of rawCounts.entries()) {
      const label = normalizeProject(raw);
      if (!label) continue;
      const existing = labelGroups.get(label);
      if (existing) {
        existing.rawValues.push(raw);
        existing.count += count;
      } else {
        labelGroups.set(label, { rawValues: [raw], count });
      }
    }

    const projects = [...labelGroups.entries()]
      .filter(([, g]) => g.count >= MIN_MEMORIES)
      .sort((a, b) => b[1].count - a[1].count)
      .map(([label, g]) => ({ label, count: g.count, rawValues: g.rawValues }));

    return new Response(
      JSON.stringify({ nodes, links, projects, entityNodes }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[memories/graph GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

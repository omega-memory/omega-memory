import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories } from "../../../../lib/db/schema";
import { desc } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * Skills manifest - defines the OMEGA skill graph.
 * In production this is imported from a shared skillsManifest module.
 * Inlined here for self-hosted portability.
 */
interface SkillDef {
  id: string;
  name: string;
  plugin: string;
  description: string;
  type: "rigid" | "flexible";
  invokes?: string[];
}

const SKILLS_MANIFEST: SkillDef[] = [
  { id: "omega_store", name: "omega_store", plugin: "memory", description: "Store a memory", type: "rigid" },
  { id: "omega_query", name: "omega_query", plugin: "memory", description: "Query memories", type: "rigid" },
  { id: "omega_reflect", name: "omega_reflect", plugin: "memory", description: "Reflect on patterns", type: "flexible", invokes: ["omega_query"] },
  { id: "omega_protocol", name: "omega_protocol", plugin: "coordination", description: "Load operating protocol", type: "rigid" },
  { id: "omega_welcome", name: "omega_welcome", plugin: "coordination", description: "Session welcome briefing", type: "rigid", invokes: ["omega_query"] },
  { id: "omega_checkpoint", name: "omega_checkpoint", plugin: "coordination", description: "Save session checkpoint", type: "rigid", invokes: ["omega_store"] },
  { id: "omega_handoff", name: "omega_handoff", plugin: "coordination", description: "Create handoff for next agent", type: "rigid", invokes: ["omega_store"] },
  { id: "omega_profile", name: "omega_profile", plugin: "identity", description: "Load user profile", type: "rigid", invokes: ["omega_query"] },
  { id: "omega_tools", name: "omega_tools", plugin: "meta", description: "List available tools", type: "rigid" },
];

const PLUGIN_COLORS: Record<string, string> = {
  memory: "#3b82f6",
  coordination: "#10b981",
  identity: "#f59e0b",
  meta: "#8b5cf6",
};

interface SkillNode {
  id: string;
  name: string;
  plugin: string;
  description: string;
  type: "rigid" | "flexible";
  usageCount: number;
  lastUsed: string | null;
}

interface SkillLink {
  source: string;
  target: string;
  type: "invocation" | "co-occurrence";
  weight: number;
}

interface PluginInfo {
  id: string;
  label: string;
  color: string;
  count: number;
}

/**
 * GET /api/admin/skills/graph?plugin=<filter>
 *
 * Returns skill nodes, invocation/co-occurrence links, and plugin info
 * for the skills force-directed graph visualization.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const url = new URL(context.request.url);
    const pluginFilter = url.searchParams.get("plugin");

    let skills = SKILLS_MANIFEST;
    if (pluginFilter) {
      skills = skills.filter((s) => s.plugin === pluginFilter);
    }

    // Query OMEGA memories for skill name mentions
    const mems = await db
      .select({
        nodeId: memories.nodeId,
        content: memories.content,
        sessionId: memories.sessionId,
        createdAt: memories.createdAt,
      })
      .from(memories)
      .orderBy(desc(memories.createdAt))
      .limit(2000);

    // Count usage per skill and track last used
    const usageCounts = new Map<string, number>();
    const lastUsedMap = new Map<string, string>();
    const sessionSkills = new Map<string, Set<string>>();

    for (const mem of mems) {
      const lower = mem.content.toLowerCase();
      for (const skill of SKILLS_MANIFEST) {
        if (lower.includes(skill.name.toLowerCase())) {
          usageCounts.set(skill.id, (usageCounts.get(skill.id) || 0) + 1);
          if (!lastUsedMap.has(skill.id) || mem.createdAt > (lastUsedMap.get(skill.id) || "")) {
            lastUsedMap.set(skill.id, mem.createdAt);
          }
          if (mem.sessionId) {
            const set = sessionSkills.get(mem.sessionId) || new Set();
            set.add(skill.id);
            sessionSkills.set(mem.sessionId, set);
          }
        }
      }
    }

    // Build nodes
    const nodes: SkillNode[] = skills.map((s) => ({
      id: s.id,
      name: s.name,
      plugin: s.plugin,
      description: s.description,
      type: s.type,
      usageCount: usageCounts.get(s.id) || 0,
      lastUsed: lastUsedMap.get(s.id) || null,
    }));

    // Build edges
    const links: SkillLink[] = [];
    const seen = new Set<string>();
    const nodeIds = new Set(nodes.map((n) => n.id));

    // Invocation edges from manifest
    for (const skill of skills) {
      if (!skill.invokes) continue;
      for (const target of skill.invokes) {
        if (!nodeIds.has(target)) continue;
        const key = `inv:${skill.id}-${target}`;
        if (!seen.has(key)) {
          links.push({ source: skill.id, target, type: "invocation", weight: 1 });
          seen.add(key);
        }
      }
    }

    // Co-occurrence edges from sessions
    const cooccurrenceCounts = new Map<string, number>();
    for (const skillSet of sessionSkills.values()) {
      const arr = [...skillSet].filter((id) => nodeIds.has(id));
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const key = [arr[i], arr[j]].sort().join("|");
          cooccurrenceCounts.set(key, (cooccurrenceCounts.get(key) || 0) + 1);
        }
      }
    }

    for (const [key, count] of cooccurrenceCounts) {
      if (count < 2) continue;
      const [a, b] = key.split("|");
      const edgeKey = `co:${a}-${b}`;
      if (!seen.has(edgeKey)) {
        links.push({ source: a, target: b, type: "co-occurrence", weight: Math.min(count / 5, 1) });
        seen.add(edgeKey);
      }
    }

    // Plugin info
    const pluginCounts = new Map<string, number>();
    for (const n of nodes) {
      pluginCounts.set(n.plugin, (pluginCounts.get(n.plugin) || 0) + 1);
    }

    const plugins: PluginInfo[] = [...pluginCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id, count]) => ({
        id,
        label: id,
        color: PLUGIN_COLORS[id] || "#505068",
        count,
      }));

    return new Response(
      JSON.stringify({ nodes, links, plugins }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[skills/graph]", msg);
    return new Response(JSON.stringify({ error: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

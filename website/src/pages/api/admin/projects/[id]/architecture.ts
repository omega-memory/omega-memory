import type { APIRoute } from "astro";
import type { InferSelectModel } from "drizzle-orm";
import { requireAdminAuth } from "../../../../../lib/api/requireAuth";
import { coordSessions, coordFileClaims } from "../../../../../lib/db/schema";
import { desc, gte } from "drizzle-orm";

type SessionRow = InferSelectModel<typeof coordSessions>;
type ClaimRow = InferSelectModel<typeof coordFileClaims>;

/**
 * GET /api/admin/projects/[id]/architecture
 *
 * Returns architecture module map, co-change edges, Mermaid syntax, and session
 * heatmap for a project. The production route queries coord_git_events and
 * coord_handoffs which don't exist in the Drizzle schema. We derive data from
 * coord_sessions and coord_file_claims.
 */

interface ArchitectureModule {
  path: string;
  fileCount: number;
  recentCommits: number;
  health: "active" | "recent" | "dormant";
}

interface ArchitectureEdge {
  from: string;
  to: string;
  weight: number;
}

interface SessionHeatmapEntry {
  date: string;
  hour: number;
  count: number;
}

/** Group a file path into its module (first 2 directory segments). */
function toModule(filePath: string): string {
  const parts = filePath.split("/").filter(Boolean);
  const start = parts.findIndex(
    (p) => !["Users", "Projects", "home", "."].includes(p),
  );
  const relevant = parts.slice(start >= 0 ? start : 0);
  return relevant.slice(0, 2).join("/") || filePath;
}

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const projectId = context.params.id;
    if (!projectId) {
      return new Response(
        JSON.stringify({
          error: "Project ID required",
          modules: [],
          edges: [],
          mermaidSyntax: "",
          sessionHeatmap: [],
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    const now = new Date();
    const days14 = new Date(now.getTime() - 14 * 86400000).toISOString();
    const days30 = new Date(now.getTime() - 30 * 86400000).toISOString();

    // Fetch sessions and file claims
    const sessions = await db
      .select()
      .from(coordSessions)
      .where(gte(coordSessions.startedAt, days14))
      .orderBy(desc(coordSessions.startedAt))
      .limit(500);

    const claims = await db
      .select()
      .from(coordFileClaims)
      .orderBy(desc(coordFileClaims.claimedAt))
      .limit(200);

    // Filter to this project (match by agentType as proxy)
    const projSessions = sessions.filter(
      (s: SessionRow) => s.agentType === projectId,
    );
    const sessionIds = new Set(projSessions.map((s: SessionRow) => s.id));
    const projClaims = claims.filter(
      (c: ClaimRow) => c.sessionId && sessionIds.has(c.sessionId),
    );

    // Build modules from file claims
    const moduleFileCount = new Map<string, Set<string>>();
    const moduleLastSeen = new Map<string, number>();

    for (const c of projClaims) {
      const mod = toModule(c.filePath);
      if (!moduleFileCount.has(mod)) moduleFileCount.set(mod, new Set());
      moduleFileCount.get(mod)!.add(c.filePath);
      const ts = c.claimedAt ? new Date(c.claimedAt).getTime() : 0;
      if (!moduleLastSeen.has(mod) || ts > moduleLastSeen.get(mod)!) {
        moduleLastSeen.set(mod, ts);
      }
    }

    const sevenDaysAgo = now.getTime() - 7 * 86400000;
    const thirtyDaysAgo = now.getTime() - 30 * 86400000;

    const modules: ArchitectureModule[] = Array.from(moduleFileCount.entries())
      .map(([path, files]) => {
        const lastSeen = moduleLastSeen.get(path) ?? 0;
        const health: ArchitectureModule["health"] =
          lastSeen >= sevenDaysAgo
            ? "active"
            : lastSeen >= thirtyDaysAgo
              ? "recent"
              : "dormant";
        return { path, fileCount: files.size, recentCommits: 0, health };
      })
      .sort((a, b) => b.fileCount - a.fileCount);

    // Build co-change edges from claims in same session
    const edgeMap = new Map<string, number>();
    const sessionFileMap = new Map<string, string[]>();

    for (const c of projClaims) {
      if (!c.sessionId) continue;
      if (!sessionFileMap.has(c.sessionId)) sessionFileMap.set(c.sessionId, []);
      sessionFileMap.get(c.sessionId)!.push(c.filePath);
    }

    for (const files of sessionFileMap.values()) {
      const mods = [...new Set(files.map(toModule))];
      for (let i = 0; i < mods.length; i++) {
        for (let j = i + 1; j < mods.length; j++) {
          const key = [mods[i], mods[j]].sort().join("|||");
          edgeMap.set(key, (edgeMap.get(key) ?? 0) + 1);
        }
      }
    }

    const edges: ArchitectureEdge[] = [];
    for (const [key, weight] of edgeMap.entries()) {
      if (weight < 2) continue;
      const [from, to] = key.split("|||");
      edges.push({ from, to, weight });
    }
    edges.sort((a, b) => b.weight - a.weight);

    // Generate Mermaid syntax
    const mermaidLines: string[] = ["flowchart TD"];

    const groups = new Map<string, ArchitectureModule[]>();
    for (const mod of modules) {
      const group = mod.path.split("/")[0];
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group)!.push(mod);
    }

    mermaidLines.push('  classDef active fill:#05966933,stroke:#059669,stroke-width:2px');
    mermaidLines.push('  classDef recent fill:#d9770633,stroke:#d97706,stroke-width:1.5px');
    mermaidLines.push('  classDef dormant fill:#64748b11,stroke:#64748b44,stroke-width:1px');

    const nodeId = (path: string) => path.replace(/[^a-zA-Z0-9]/g, "_");

    for (const [group, mods] of groups.entries()) {
      if (mods.length === 1 && groups.size > 1) {
        const mod = mods[0];
        const id = nodeId(mod.path);
        const label = mod.path.split("/").pop() ?? mod.path;
        mermaidLines.push(`  ${id}["${label}"]:::${mod.health}`);
      } else {
        mermaidLines.push(`  subgraph ${nodeId(group)}["${group}"]`);
        for (const mod of mods) {
          const id = nodeId(mod.path);
          const label = mod.path.split("/").slice(1).join("/") || mod.path;
          mermaidLines.push(`    ${id}["${label}"]:::${mod.health}`);
        }
        mermaidLines.push("  end");
      }
    }

    for (const edge of edges.slice(0, 15)) {
      const fromId = nodeId(edge.from);
      const toId = nodeId(edge.to);
      if (edge.weight >= 5) {
        mermaidLines.push(`  ${fromId} ==> ${toId}`);
      } else {
        mermaidLines.push(`  ${fromId} --> ${toId}`);
      }
    }

    const mermaidSyntax = mermaidLines.join("\n");

    // Session heatmap (14 days x 24 hours)
    const sessionHeatmap: SessionHeatmapEntry[] = [];
    const heatmapMap = new Map<string, number>();

    for (const s of projSessions) {
      if (!s.startedAt) continue;
      const d = new Date(s.startedAt);
      const dateStr = d.toISOString().slice(0, 10);
      const hour = d.getUTCHours();
      const key = `${dateStr}:${hour}`;
      heatmapMap.set(key, (heatmapMap.get(key) ?? 0) + 1);
    }

    for (const [key, count] of heatmapMap.entries()) {
      const [date, hourStr] = key.split(":");
      sessionHeatmap.push({ date, hour: parseInt(hourStr, 10), count });
    }

    return new Response(
      JSON.stringify({ modules, edges, mermaidSyntax, sessionHeatmap }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "private, no-store",
        },
      },
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[projects/architecture]", msg);
    return new Response(
      JSON.stringify({
        error: msg,
        modules: [],
        edges: [],
        mermaidSyntax: "",
        sessionHeatmap: [],
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
};

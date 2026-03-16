import type { APIRoute } from "astro";
import type { InferSelectModel } from "drizzle-orm";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { coordSessions, coordFileClaims } from "../../../../lib/db/schema";
import { desc, gte } from "drizzle-orm";

type ClaimRow = InferSelectModel<typeof coordFileClaims>;

/**
 * GET /api/admin/projects/overview
 *
 * Returns project overview data derived from coord_sessions and coord_file_claims.
 * The production route queries projects, coord_sessions, coord_tasks, coord_decisions,
 * coord_handoffs, coord_git_events, coord_file_claims — most of which don't exist
 * in the Drizzle schema. We return graceful data from what's available.
 */

type ProjectHealth = "blocked" | "attention" | "healthy" | "stale";

function computeHealth(lastActive: string | null): ProjectHealth {
  if (!lastActive) return "stale";
  const daysSince = Math.floor(
    (Date.now() - new Date(lastActive).getTime()) / (1000 * 60 * 60 * 24),
  );
  if (daysSince >= 14) return "stale";
  if (daysSince >= 5) return "attention";
  return "healthy";
}

function buildSparkline(
  sessions: { startedAt: string | null }[],
): number[] {
  const now = new Date();
  const buckets = new Array(14).fill(0);
  for (const s of sessions) {
    if (!s.startedAt) continue;
    const age = now.getTime() - new Date(s.startedAt).getTime();
    const dayIndex = Math.floor(age / (1000 * 60 * 60 * 24));
    if (dayIndex >= 0 && dayIndex < 14) {
      buckets[13 - dayIndex]++;
    }
  }
  return buckets;
}

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const now = new Date();
    const days7 = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
    const days14 = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000).toISOString();
    const days30 = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();

    // Fetch sessions from last 30 days
    const sessions = await db
      .select()
      .from(coordSessions)
      .where(gte(coordSessions.startedAt, days30))
      .orderBy(desc(coordSessions.startedAt))
      .limit(2000);

    // Fetch recent file claims
    const claims = await db
      .select()
      .from(coordFileClaims)
      .where(gte(coordFileClaims.claimedAt, days30))
      .orderBy(desc(coordFileClaims.claimedAt))
      .limit(200);

    // Group sessions by agentType as proxy for project
    type SessionRow = (typeof sessions)[number];
    const projectMap = new Map<string, SessionRow[]>();
    for (const s of sessions) {
      const key = s.agentType ?? "unknown";
      if (!projectMap.has(key)) projectMap.set(key, []);
      projectMap.get(key)!.push(s);
    }

    interface AttentionItem {
      projectId: string;
      projectName: string;
      severity: "red" | "amber";
      reason: string;
    }
    const needsAttention: AttentionItem[] = [];

    const projects = Array.from(projectMap.entries()).map(([name, projSessions]) => {
      const projSessions7d = projSessions.filter(
        (s) => s.startedAt && new Date(s.startedAt).getTime() >= new Date(days7).getTime(),
      );
      const projSessions14d = projSessions.filter(
        (s) => s.startedAt && new Date(s.startedAt).getTime() >= new Date(days14).getTime(),
      );

      const lastActive = projSessions[0]?.startedAt ?? null;
      const hasActiveSession = projSessions.some((s) => s.status === "active");
      const health = computeHealth(lastActive);
      const sparkline = buildSparkline(projSessions14d);

      // File claims for this project's sessions
      const sessionIds = new Set(projSessions.map((s) => s.id));
      const projClaims = claims
        .filter((c: ClaimRow) => c.sessionId && sessionIds.has(c.sessionId))
        .slice(0, 15)
        .map((c: ClaimRow) => ({
          filePath: c.filePath,
          sessionId: c.sessionId,
          claimedAt: c.claimedAt,
        }));

      if (health === "stale") {
        const daysSince = lastActive
          ? Math.floor((Date.now() - new Date(lastActive).getTime()) / (1000 * 60 * 60 * 24))
          : 999;
        needsAttention.push({
          projectId: name,
          projectName: name,
          severity: "amber",
          reason: `No activity in ${daysSince}d`,
        });
      }

      return {
        id: name,
        name,
        category: "Tool",
        health,
        lastActive,
        hasActiveSession,
        summary: null,
        gitBranch: null,
        sessionCount7d: projSessions7d.length,
        sessionCount30d: projSessions.length,
        commitCount30d: 0,
        decisionCount30d: 0,
        launchProgress: 0,
        tasks: { pending: 0, inProgress: 0, completed: 0, failed: 0, blocked: 0, items: [] },
        sparkline,
        latestHandoff: null,
        decisions: [],
        fileClaims: projClaims,
        activity: [],
      };
    });

    // Sort: blocked first, then by recency
    const healthOrder: Record<ProjectHealth, number> = {
      blocked: 0,
      attention: 1,
      healthy: 2,
      stale: 3,
    };
    projects.sort((a, b) => {
      const ha = healthOrder[a.health];
      const hb = healthOrder[b.health];
      if (ha !== hb) return ha - hb;
      if (!a.lastActive && !b.lastActive) return 0;
      if (!a.lastActive) return 1;
      if (!b.lastActive) return -1;
      return new Date(b.lastActive).getTime() - new Date(a.lastActive).getTime();
    });

    needsAttention.sort((a, b) => {
      if (a.severity === b.severity) return 0;
      return a.severity === "red" ? -1 : 1;
    });

    return new Response(
      JSON.stringify({
        projects,
        needsAttention,
        generatedAt: now.toISOString(),
      }),
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
    console.error("[projects/overview]", msg);
    return new Response(
      JSON.stringify({
        error: msg,
        projects: [],
        needsAttention: [],
        generatedAt: new Date().toISOString(),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

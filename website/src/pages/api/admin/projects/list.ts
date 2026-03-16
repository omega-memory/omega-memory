import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { coordSessions, coordFileClaims } from "../../../../lib/db/schema";
import { desc, eq, inArray } from "drizzle-orm";

/**
 * GET /api/admin/projects/list
 *
 * Returns a lightweight list of projects derived from coord_sessions.
 * The production route queries a "projects" table + coord_sessions + coord_tasks;
 * since those tables don't exist in the Drizzle schema, we derive project names
 * from coord_sessions and return graceful data.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // Fetch recent sessions grouped by project-like fields
    const sessions = await db
      .select()
      .from(coordSessions)
      .orderBy(desc(coordSessions.startedAt))
      .limit(2000);

    // Derive unique "projects" from agentType or metadata
    const projectMap = new Map<
      string,
      {
        lastActive: string | null;
        sessionCount: number;
        hasActiveSession: boolean;
      }
    >();

    for (const s of sessions) {
      const projectKey = s.agentType ?? "unknown";
      const existing = projectMap.get(projectKey);
      if (!existing) {
        projectMap.set(projectKey, {
          lastActive: s.startedAt,
          sessionCount: 1,
          hasActiveSession: s.status === "active",
        });
      } else {
        existing.sessionCount++;
        if (s.status === "active") existing.hasActiveSession = true;
        if (
          s.startedAt &&
          (!existing.lastActive || s.startedAt > existing.lastActive)
        ) {
          existing.lastActive = s.startedAt;
        }
      }
    }

    const projects = Array.from(projectMap.entries())
      .map(([name, info]) => ({
        id: name,
        name,
        category: "tool",
        lastActive: info.lastActive,
        sessionCount: info.sessionCount,
        activeTaskCount: 0,
        hasActiveSession: info.hasActiveSession,
      }))
      .sort((a, b) => {
        if (!a.lastActive && !b.lastActive) return 0;
        if (!a.lastActive) return 1;
        if (!b.lastActive) return -1;
        return (
          new Date(b.lastActive).getTime() - new Date(a.lastActive).getTime()
        );
      });

    return new Response(JSON.stringify({ projects }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "private, no-store",
      },
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[projects/list]", msg);
    return new Response(JSON.stringify({ error: msg, projects: [] }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

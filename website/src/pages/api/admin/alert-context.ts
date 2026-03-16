import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories, coordSessions, coordFileClaims, scheduleRuns, schedules } from "../../../lib/db/schema";
import { eq, gte, desc, and, ne, count } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json" };

/**
 * GET /api/admin/alert-context?type=<alertType>&param=<optional>
 *
 * Lazy-loads contextual data for a specific alert type.
 * Called on-demand when user expands an alert card.
 *
 * Some production alert types (failed_post, engagement_declining) reference
 * tables not in the self-hosted schema (tweet_queue, tweet_performance).
 * Those return graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const type = url.searchParams.get("type");

  if (!type) {
    return new Response(JSON.stringify({ error: "type parameter required" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  try {
    switch (type) {
      case "failing_job": {
        const scheduleId = url.searchParams.get("job");
        if (!scheduleId) {
          return new Response(JSON.stringify({ error: "job parameter required" }), {
            status: 400,
            headers: JSON_HEADERS,
          });
        }

        const runs = await db
          .select({
            status: scheduleRuns.status,
            startedAt: scheduleRuns.startedAt,
            error: scheduleRuns.error,
          })
          .from(scheduleRuns)
          .where(eq(scheduleRuns.scheduleId, scheduleId))
          .orderBy(desc(scheduleRuns.startedAt))
          .limit(3);

        const schedule = await db
          .select()
          .from(schedules)
          .where(eq(schedules.id, scheduleId))
          .limit(1);

        return new Response(
          JSON.stringify({
            type: "failing_job",
            title: `Job failing: ${schedule[0]?.name ?? scheduleId}`,
            severity: "critical",
            timestamp: new Date().toISOString(),
            detail: {
              jobName: schedule[0]?.name ?? scheduleId,
              lastError: runs[0]?.error ?? null,
              recentRuns: runs.map((r: { status: string | null; startedAt: string | null; error: string | null }) => ({
                status: r.status,
                startedAt: r.startedAt,
                error: r.error ?? null,
              })),
            },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      case "failed_post": {
        // tweet_queue table not in self-hosted schema
        return new Response(
          JSON.stringify({
            type: "failed_post",
            title: "No failed posts data available",
            severity: "info",
            timestamp: new Date().toISOString(),
            detail: { failedCount: 0, recentFailed: [], recentSuccessful: [] },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      case "cloud_sync_gap": {
        // In self-hosted, count total memories as proxy
        const totalRes = await db.select({ value: count() }).from(memories);
        const total = totalRes[0]?.value ?? 0;

        return new Response(
          JSON.stringify({
            type: "cloud_sync_gap",
            title: `Cloud sync: ${total} memories`,
            severity: "info",
            timestamp: new Date().toISOString(),
            detail: {
              localCount: total,
              cloudCount: 0,
              gapPct: 100,
              lastSyncAt: null,
              unsyncedCount: total,
            },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      case "engagement_declining": {
        // tweet_performance table not in self-hosted schema
        return new Response(
          JSON.stringify({
            type: "engagement_declining",
            title: "No engagement data available",
            severity: "info",
            timestamp: new Date().toISOString(),
            detail: { currentRate: 0, previousRate: 0, recentPosts: [] },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      case "memory_spike": {
        const spikeCutoff = new Date(Date.now() - 60 * 60_000).toISOString();
        const recent = await db
          .select({
            content: memories.content,
            memoryType: memories.memoryType,
            metadata: memories.metadata,
            createdAt: memories.createdAt,
          })
          .from(memories)
          .where(gte(memories.createdAt, spikeCutoff))
          .orderBy(desc(memories.createdAt))
          .limit(5);

        const countRes = await db
          .select({ value: count() })
          .from(memories)
          .where(gte(memories.createdAt, spikeCutoff));
        const totalInLastHour = countRes[0]?.value ?? 0;

        return new Response(
          JSON.stringify({
            type: "memory_spike",
            title: `Memory spike: ${totalInLastHour} in last hour`,
            severity: "info",
            timestamp: new Date().toISOString(),
            detail: {
              recentMemories: recent.map((m: { content: string; memoryType: string | null; metadata: string | null; createdAt: string }) => {
                let meta: Record<string, unknown> = {};
                try { meta = m.metadata ? JSON.parse(m.metadata) : {}; } catch { /* ignore */ }
                return {
                  content: (m.content ?? "").slice(0, 100),
                  memoryType: m.memoryType ?? "unknown",
                  agentId: (meta.session_id as string) ?? null,
                  createdAt: m.createdAt,
                };
              }),
              totalInLastHour,
            },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      case "coordination_conflict": {
        const activeCutoff = new Date(Date.now() - 2 * 60_000).toISOString();

        const sessions = await db
          .select()
          .from(coordSessions)
          .where(and(ne(coordSessions.status, "ended")));

        const claims = await db.select().from(coordFileClaims);

        const activeIds = new Set(sessions.map((s: typeof sessions[number]) => s.id));
        const sessionMap = new Map(sessions.map((s: typeof sessions[number]) => [s.id, s] as const));

        const fileSessions = new Map<string, string[]>();
        for (const c of claims) {
          if (activeIds.has(c.sessionId || "")) {
            const arr = fileSessions.get(c.filePath) ?? [];
            arr.push(c.sessionId || "");
            fileSessions.set(c.filePath, arr);
          }
        }

        const conflicts = [...fileSessions.entries()]
          .filter(([, sids]) => sids.length > 1)
          .map(([path, sids]) => ({
            filePath: path,
            sessions: sids.map((sid) => {
              const s = sessionMap.get(sid);
              return {
                sessionId: sid,
                project: null,
                intent: null,
              };
            }),
          }));

        return new Response(
          JSON.stringify({
            type: "coordination_conflict",
            title: `${conflicts.length} file conflict(s)`,
            severity: conflicts.length > 0 ? "critical" : "info",
            timestamp: new Date().toISOString(),
            detail: { conflicts },
          }),
          { status: 200, headers: JSON_HEADERS },
        );
      }

      default:
        return new Response(JSON.stringify({ error: `Unknown alert type: ${type}` }), {
          status: 400,
          headers: JSON_HEADERS,
        });
    }
  } catch (err: unknown) {
    console.error("[alert-context]", err);
    return new Response(JSON.stringify({ error: "Internal error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

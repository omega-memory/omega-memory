import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import {
  memories,
  coordSessions,
  coordFileClaims,
  adminAlerts,
} from "../../../lib/db/schema";
import { eq, gte, and, ne, inArray } from "drizzle-orm";

/**
 * GET /api/admin/ambient-status
 *
 * Lightweight ambient status endpoint for sidebar badges, tab title, and toasts.
 * Returns only counts and flags. Designed for 30s polling.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const now = Date.now();

    // Active = heartbeat within last 2 minutes
    const activeCutoff = new Date(now - 2 * 60_000).toISOString();
    // Recent memories = last hour
    const memoryCutoff = new Date(now - 60 * 60_000).toISOString();
    // Sessions shown = heartbeat within last 12h (matches coordination tab)
    const sessionCutoff = new Date(now - 12 * 60 * 60_000).toISOString();
    // 24h cutoff for sparkline
    const sparklineCutoff = new Date(now - 24 * 60 * 60_000).toISOString();

    // Fetch sessions, file claims, recent memories, and sparkline data in parallel
    const [sessions, claims, recentMemoryRows, sparklineRows] = await Promise.all([
      db
        .select({
          id: coordSessions.id,
          status: coordSessions.status,
          startedAt: coordSessions.startedAt,
        })
        .from(coordSessions)
        .where(
          and(
            ne(coordSessions.status, "ended"),
            gte(coordSessions.startedAt, sessionCutoff),
          ),
        ),
      db
        .select({
          filePath: coordFileClaims.filePath,
          sessionId: coordFileClaims.sessionId,
        })
        .from(coordFileClaims)
        .where(gte(coordFileClaims.claimedAt, sessionCutoff)),
      db
        .select({ id: memories.id })
        .from(memories)
        .where(
          and(
            eq(memories.userId, user.id),
            gte(memories.createdAt, memoryCutoff),
          ),
        ),
      db
        .select({ createdAt: memories.createdAt })
        .from(memories)
        .where(
          and(
            eq(memories.userId, user.id),
            gte(memories.createdAt, sparklineCutoff),
          ),
        ),
    ]);

    // Count active agents (started within last 2 minutes as proxy for heartbeat)
    const activeAgents = sessions.filter(
      (s: any) => s.startedAt && new Date(s.startedAt).getTime() > new Date(activeCutoff).getTime(),
    ).length;

    // Detect file conflicts: files claimed by 2+ active sessions
    const activeSessionIds = new Set(
      sessions
        .filter((s: any) => s.startedAt && new Date(s.startedAt).getTime() > new Date(activeCutoff).getTime())
        .map((s: any) => s.id),
    );
    const fileSessions = new Map<string, string[]>();
    for (const c of claims) {
      if (c.sessionId && activeSessionIds.has(c.sessionId)) {
        const arr = fileSessions.get(c.filePath) ?? [];
        arr.push(c.sessionId);
        fileSessions.set(c.filePath, arr);
      }
    }
    const fileConflicts = [...fileSessions.entries()]
      .filter(([, sids]) => sids.length > 1)
      .map(([path]) => path);

    const recentMemoryCount = recentMemoryRows.length;

    // Build 24-hour sparkline: 24 buckets, most recent last
    const sparkline = new Array(24).fill(0);
    const sparklineStart = now - 24 * 60 * 60_000;
    for (const row of sparklineRows) {
      if (row.createdAt) {
        const t = new Date(row.createdAt).getTime();
        const bucket = Math.min(23, Math.floor((t - sparklineStart) / (60 * 60_000)));
        sparkline[bucket]++;
      }
    }

    // Persist alerts (deduplicated by type + 1h window)
    const alertsToCheck: { type: string; severity: string; message: string; metadata: Record<string, unknown> }[] = [];

    if (fileConflicts.length > 0) {
      alertsToCheck.push({
        type: "coordination_conflict",
        severity: "critical",
        message: `${fileConflicts.length} file conflict(s)`,
        metadata: { files: fileConflicts },
      });
    }

    if (recentMemoryCount > 50) {
      alertsToCheck.push({
        type: "memory_spike",
        severity: "info",
        message: `Memory spike: ${recentMemoryCount} in last hour`,
        metadata: { count: recentMemoryCount },
      });
    }

    // Persist new alerts (skip if same type exists within last hour)
    if (alertsToCheck.length > 0) {
      const oneHourAgo = new Date(now - 60 * 60_000).toISOString();
      const alertTypes = alertsToCheck.map((a) => a.type);
      const existingAlerts = await db
        .select({ type: adminAlerts.type })
        .from(adminAlerts)
        .where(
          and(
            inArray(adminAlerts.type, alertTypes),
            gte(adminAlerts.createdAt, oneHourAgo),
          ),
        );

      const existingTypes = new Set(existingAlerts.map((a: any) => a.type));

      for (const alert of alertsToCheck) {
        if (!existingTypes.has(alert.type)) {
          await db.insert(adminAlerts).values({
            id: crypto.randomUUID(),
            type: alert.type,
            severity: alert.severity,
            message: alert.message,
            metadata: JSON.stringify(alert.metadata),
            status: "active",
            createdAt: new Date().toISOString(),
          });
        }
      }
    }

    // Auto-resolve: if no file conflicts, resolve any active conflict alerts
    if (fileConflicts.length === 0) {
      await db
        .update(adminAlerts)
        .set({ status: "resolved", resolvedAt: new Date().toISOString() })
        .where(
          and(
            eq(adminAlerts.type, "coordination_conflict"),
            eq(adminAlerts.status, "active"),
          ),
        );
    }

    return new Response(
      JSON.stringify({
        activeAgents,
        totalSessions: sessions.length,
        recentMemories: recentMemoryCount,
        fileConflicts,
        sparkline,
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        },
      },
    );
  } catch (err: unknown) {
    console.error("[ambient-status] unexpected error:", err);
    return new Response(
      JSON.stringify({
        error: "Database error",
        activeAgents: 0,
        totalSessions: 0,
        recentMemories: 0,
        fileConflicts: [],
        sparkline: [],
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};

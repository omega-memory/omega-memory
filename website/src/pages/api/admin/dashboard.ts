import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db";
import {
  memories,
  coordSessions,
  syncState,
  adminAlerts,
} from "../../../lib/db/schema";
import {
  AuthEngine,
  getSessionFromCookie,
} from "../../../lib/auth";
import { count, desc, eq } from "drizzle-orm";

export const GET: APIRoute = async (context) => {
  // Auth check
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);
  if (!token) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const db = getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "dev-secret";
  const auth = new AuthEngine(db, secret);
  const user = await auth.validateSession(token);
  if (!user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Gather metrics
  const [memoryCount] = await db.select({ value: count() }).from(memories);

  const [activeSessionCount] = await db
    .select({ value: count() })
    .from(coordSessions)
    .where(eq(coordSessions.status, "active"));

  const [latestSync] = await db
    .select()
    .from(syncState)
    .orderBy(desc(syncState.lastSyncAt))
    .limit(1);

  const recentMemories = await db
    .select({
      id: memories.id,
      nodeId: memories.nodeId,
      content: memories.content,
      eventType: memories.eventType,
      project: memories.project,
      createdAt: memories.createdAt,
      status: memories.status,
    })
    .from(memories)
    .orderBy(desc(memories.createdAt))
    .limit(10);

  const alerts = await db
    .select()
    .from(adminAlerts)
    .where(eq(adminAlerts.status, "active"))
    .orderBy(desc(adminAlerts.createdAt))
    .limit(10);

  const syncOk = latestSync?.lastSyncAt
    ? Date.now() - new Date(latestSync.lastSyncAt).getTime() < 5 * 60 * 1000
    : false;

  return new Response(
    JSON.stringify({
      memoryCount: memoryCount.value,
      activeSessions: activeSessionCount.value,
      syncStatus: syncOk ? "synced" : "stale",
      lastSyncAt: latestSync?.lastSyncAt ?? null,
      recentMemories,
      alerts,
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }
  );
};

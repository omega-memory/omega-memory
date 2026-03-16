import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import {
  memories,
  coordSessions,
  syncState,
  adminAlerts,
  schedules,
} from "../../../lib/db/schema";
import { count, desc, eq, gte, like, or, sql } from "drizzle-orm";

// ── GitHub (public, no auth) ────────────────────────────────────

async function fetchGitHub() {
  try {
    const [repoRes, contribRes] = await Promise.all([
      fetch("https://api.github.com/repos/omega-memory/omega-memory", {
        headers: { Accept: "application/vnd.github.v3+json" },
      }),
      fetch(
        "https://api.github.com/repos/omega-memory/omega-memory/contributors?per_page=1&anon=1",
        { headers: { Accept: "application/vnd.github.v3+json" } },
      ),
    ]);

    if (!repoRes.ok) return null;
    const repo = await repoRes.json();

    let contributors = 1;
    if (contribRes.ok) {
      const link = contribRes.headers.get("link");
      if (link) {
        const match = link.match(/page=(\d+)>; rel="last"/);
        if (match) contributors = parseInt(match[1], 10);
      } else {
        const list = await contribRes.json();
        contributors = Array.isArray(list) ? list.length : 1;
      }
    }

    return {
      stars: repo.stargazers_count as number,
      forks: repo.forks_count as number,
      openIssues: repo.open_issues_count as number,
      pushedAt: repo.pushed_at as string,
      contributors,
      watchers: repo.subscribers_count as number,
    };
  } catch {
    return null;
  }
}

// ── PyPI ────────────────────────────────────────────────────────

async function fetchPyPI() {
  try {
    const [metaRes, statsRes] = await Promise.all([
      fetch("https://pypi.org/pypi/omega-memory/json"),
      fetch("https://pypistats.org/api/packages/omega-memory/recent"),
    ]);

    if (!metaRes.ok) return null;
    const meta = await metaRes.json();

    let monthlyDownloads = 0;
    let weeklyDownloads = 0;
    if (statsRes.ok) {
      const stats = await statsRes.json();
      monthlyDownloads = stats.data?.last_month ?? 0;
      weeklyDownloads = stats.data?.last_week ?? 0;
    }

    return {
      version: (meta.info?.version as string) ?? "unknown",
      monthlyDownloads,
      weeklyDownloads,
    };
  } catch {
    return null;
  }
}

// ── Grants (memory search) ──────────────────────────────────────

async function fetchGrants(db: any, userId: string) {
  try {
    const keywords = ["grant", "NSF", "NLnet", "SBIR", "funding", "CZI", "EOSS"];
    const conditions = keywords.map((k) => like(memories.content, `%${k}%`));

    const rows = await db
      .select({
        id: memories.id,
        content: memories.content,
        eventType: memories.eventType,
        createdAt: memories.createdAt,
        project: memories.project,
      })
      .from(memories)
      .where(or(...conditions))
      .orderBy(desc(memories.createdAt))
      .limit(50);

    return rows.map((m: any) => {
      const content = (m.content || "").toLowerCase();
      let inferredStage = "research";
      if (content.includes("submit") || content.includes("submitted"))
        inferredStage = "submitted";
      else if (content.includes("under review") || content.includes("reviewing"))
        inferredStage = "under_review";
      else if (content.includes("decided") || content.includes("awarded") || content.includes("declined"))
        inferredStage = "decided";
      else if (content.includes("prepare") || content.includes("writing"))
        inferredStage = "prepare";
      else if (content.includes("qualify") || content.includes("eligible"))
        inferredStage = "qualify";

      return {
        id: m.id,
        content: m.content,
        type: m.eventType,
        createdAt: m.createdAt,
        inferredStage,
        project: m.project || "general",
      };
    });
  } catch {
    return [];
  }
}

// ── Handler ─────────────────────────────────────────────────────

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // Gather local DB metrics + external APIs in parallel
    const [
      memoryCountRes,
      activeSessionCountRes,
      latestSyncRes,
      recentMemoriesRes,
      alertsRes,
      schedulesRes,
      github,
      pypi,
      grants,
    ] = await Promise.allSettled([
      db.select({ value: count() }).from(memories),
      db.select({ value: count() }).from(coordSessions).where(eq(coordSessions.status, "active")),
      db.select().from(syncState).orderBy(desc(syncState.lastSyncAt)).limit(1),
      db
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
        .limit(10),
      db
        .select()
        .from(adminAlerts)
        .where(eq(adminAlerts.status, "active"))
        .orderBy(desc(adminAlerts.createdAt))
        .limit(10),
      db.select().from(schedules).orderBy(schedules.name),
      fetchGitHub(),
      fetchPyPI(),
      fetchGrants(db, user.id),
    ]);

    const memoryCount =
      memoryCountRes.status === "fulfilled" ? memoryCountRes.value[0]?.value ?? 0 : 0;
    const activeSessions =
      activeSessionCountRes.status === "fulfilled"
        ? activeSessionCountRes.value[0]?.value ?? 0
        : 0;
    const latestSync =
      latestSyncRes.status === "fulfilled" ? latestSyncRes.value[0] ?? null : null;
    const recentMemories =
      recentMemoriesRes.status === "fulfilled" ? recentMemoriesRes.value : [];
    const alerts = alertsRes.status === "fulfilled" ? alertsRes.value : [];
    const schedulesList = schedulesRes.status === "fulfilled" ? schedulesRes.value : [];

    const syncOk = latestSync?.lastSyncAt
      ? Date.now() - new Date(latestSync.lastSyncAt).getTime() < 5 * 60 * 1000
      : false;

    return new Response(
      JSON.stringify({
        // Original local dashboard shape
        memoryCount,
        activeSessions,
        syncStatus: syncOk ? "synced" : "stale",
        lastSyncAt: latestSync?.lastSyncAt ?? null,
        recentMemories,
        alerts,
        // Production dashboard shape (external APIs)
        github: github.status === "fulfilled" ? github.value : null,
        pypi: pypi.status === "fulfilled" ? pypi.value : null,
        twitter: null, // Twitter/X requires OAuth keys — return null when not configured
        outreach: null, // Requires engagement_suggestions table (not in self-hosted schema)
        grants: grants.status === "fulfilled" ? grants.value : null,
        performance: null, // Requires external metrics lib
        downloads: null, // Requires download_events table
        schedules: schedulesList,
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
    console.error("[dashboard GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

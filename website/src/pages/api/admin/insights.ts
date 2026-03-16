import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import {
  memories,
  coordSessions,
  coordFileClaims,
  schedules,
  scheduleRuns,
} from "../../../lib/db/schema";
import { count, desc, eq, gte, lt, and, inArray, sql } from "drizzle-orm";

// ── Signal types (high-value memory types) ──────────────────────

const SIGNAL_TYPES = new Set([
  "decision",
  "lesson_learned",
  "preference",
  "user_preference",
  "architecture",
  "requirement",
  "key_insight",
  "outcome",
]);

// ── Insight types ───────────────────────────────────────────────

interface MemoryInsight {
  severity: "warning" | "positive" | "info";
  headline: string;
  detail: string;
  suggestion?: string;
}

// ── Compute memory insights ─────────────────────────────────────

function computeMemoryInsights({
  byType,
  total,
  activeSessions,
  growthPct,
  projects,
  unscopedCount,
  priorities,
}: {
  byType: Record<string, number>;
  total: number;
  activeSessions: number;
  growthPct: number;
  projects: { displayName: string; totalMemories: number }[];
  unscopedCount: number;
  priorities: number[];
}): MemoryInsight[] {
  const insights: MemoryInsight[] = [];

  // 1. Signal-to-noise ratio
  if (total > 0) {
    let signalCount = 0;
    for (const [type, count] of Object.entries(byType)) {
      if (SIGNAL_TYPES.has(type)) signalCount += count;
    }
    const signalPct = Math.round((signalCount / total) * 100);
    if (signalPct > 80) {
      insights.push({
        severity: "positive",
        headline: `${signalPct}% signal ratio`,
        detail: `${signalCount} of ${total} memories are high-value types (decisions, lessons, preferences).`,
      });
    } else if (signalPct < 50) {
      insights.push({
        severity: "warning",
        headline: `Low signal ratio: ${signalPct}%`,
        detail: `Only ${signalCount} of ${total} memories are high-value types. The rest is coordination noise.`,
        suggestion: "Review what your agents are storing. Consider raising capture gates.",
      });
    }
  }

  // 2. Type dominance
  if (total > 10) {
    for (const [type, count] of Object.entries(byType)) {
      const pct = Math.round((count / total) * 100);
      if (pct > 40) {
        insights.push({
          severity: "info",
          headline: `${type.replace(/_/g, " ")} dominates at ${pct}%`,
          detail: `${count} of ${total} memories are ${type.replace(/_/g, " ")}. This may indicate a healthy focus or a capture imbalance.`,
        });
        break;
      }
    }
  }

  // 3. Session productivity
  if (activeSessions > 0) {
    const perSession = total / activeSessions;
    if (perSession < 2) {
      insights.push({
        severity: "warning",
        headline: `Low capture: ${perSession.toFixed(1)}/session`,
        detail: `${total} memories across ${activeSessions} sessions. Most sessions produce little durable memory.`,
        suggestion: "Check that agents are storing decisions, not just reading.",
      });
    } else if (perSession >= 5) {
      insights.push({
        severity: "positive",
        headline: `Strong capture: ${perSession.toFixed(1)}/session`,
        detail: `${total} memories across ${activeSessions} sessions. Each session contributes meaningfully.`,
      });
    }
  }

  // 4. Growth trend
  if (growthPct !== -1) {
    if (growthPct < -30) {
      insights.push({
        severity: "warning",
        headline: `Volume down ${Math.abs(growthPct)}%`,
        detail: "Memory capture has dropped significantly compared to the previous period.",
        suggestion: "Check if agents are running less often or if capture gates are too strict.",
      });
    } else if (growthPct > 50) {
      insights.push({
        severity: "positive",
        headline: `Volume up ${growthPct}%`,
        detail: "Memory capture has grown substantially compared to the previous period.",
      });
    } else {
      insights.push({
        severity: "info",
        headline: `Volume ${growthPct >= 0 ? "+" : ""}${growthPct}% vs prior period`,
        detail: "Memory capture is roughly stable compared to the previous period.",
      });
    }
  }

  // 5. Project concentration
  const projectScoped = projects.filter((p) => p.totalMemories > 0);
  if (projectScoped.length > 1) {
    const projectTotal = projectScoped.reduce((a, p) => a + p.totalMemories, 0);
    if (projectTotal > 10) {
      const top = projectScoped.sort((a, b) => b.totalMemories - a.totalMemories)[0];
      const topPct = Math.round((top.totalMemories / projectTotal) * 100);
      if (topPct > 70) {
        insights.push({
          severity: "info",
          headline: `${top.displayName} is ${topPct}% of project memories`,
          detail: `${top.totalMemories} of ${projectTotal} project-scoped memories. Other projects may need more attention.`,
        });
      }
    }
  }

  // 6. Unscoped memories
  if (total > 0 && unscopedCount > 5) {
    const unscopedPct = Math.round((unscopedCount / total) * 100);
    if (unscopedPct > 10) {
      insights.push({
        severity: "warning",
        headline: `${unscopedPct}% unscoped (${unscopedCount})`,
        detail: "These memories lack project context, making them harder to retrieve in context.",
        suggestion: "Ensure agents run from project directories so memories auto-scope.",
      });
    }
  }

  // 7. Priority distribution
  if (priorities.length > 10) {
    const lowCount = priorities.filter((p) => p <= 2).length;
    const highCount = priorities.filter((p) => p >= 4).length;
    const lowPct = Math.round((lowCount / priorities.length) * 100);
    const highPct = Math.round((highCount / priorities.length) * 100);
    if (lowPct > 40) {
      insights.push({
        severity: "warning",
        headline: `${lowPct}% low-priority memories`,
        detail: `${lowCount} of ${priorities.length} memories are priority 1-2. This dilutes retrieval quality.`,
        suggestion: "Raise the minimum capture priority or add stricter gates.",
      });
    } else if (highPct > 60) {
      insights.push({
        severity: "positive",
        headline: `${highPct}% high-priority memories`,
        detail: `${highCount} of ${priorities.length} memories are priority 4-5. Retrieval should be high quality.`,
      });
    }
  }

  // Rank: warnings first, then positive, then info. Return top 5.
  const severityOrder: Record<string, number> = { warning: 0, positive: 1, info: 2 };
  insights.sort((a, b) => (severityOrder[a.severity] ?? 2) - (severityOrder[b.severity] ?? 2));
  return insights.slice(0, 5);
}

// ── Route handler ───────────────────────────────────────────────

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const url = new URL(context.request.url);
    const period = Math.min(
      90,
      Math.max(7, parseInt(url.searchParams.get("period") || "30", 10) || 30),
    );

    const userId = user.id;
    const now = new Date();
    const cutoff = new Date(now.getTime() - period * 86400000).toISOString();
    const prevCutoff = new Date(now.getTime() - period * 2 * 86400000).toISOString();
    const heatmapCutoff = new Date(now.getTime() - 84 * 86400000).toISOString();
    const earliestCutoff = cutoff < heatmapCutoff ? cutoff : heatmapCutoff;

    // Fetch all memories from the earliest cutoff (covers both period and heatmap)
    const allMemories = await db
      .select({
        eventType: memories.eventType,
        project: memories.project,
        createdAt: memories.createdAt,
        metadata: memories.metadata,
        priority: memories.priority,
      })
      .from(memories)
      .where(
        and(
          eq(memories.userId, userId),
          gte(memories.createdAt, earliestCutoff),
        ),
      )
      .orderBy(desc(memories.createdAt));

    // Previous period count
    const prevMemories = await db
      .select({ value: count() })
      .from(memories)
      .where(
        and(
          eq(memories.userId, userId),
          gte(memories.createdAt, prevCutoff),
          lt(memories.createdAt, cutoff),
        ),
      );
    const prevCount = prevMemories[0]?.value ?? 0;

    // Total all-time count
    const totalAllRes = await db
      .select({ value: count() })
      .from(memories)
      .where(eq(memories.userId, userId));
    const totalAll = totalAllRes[0]?.value ?? 0;

    // Highlights: recent decisions and lessons
    const highlights = await db
      .select({
        content: memories.content,
        eventType: memories.eventType,
        project: memories.project,
        createdAt: memories.createdAt,
      })
      .from(memories)
      .where(
        and(
          eq(memories.userId, userId),
          inArray(memories.eventType, ["decision", "lesson_learned"]),
          gte(memories.createdAt, cutoff),
        ),
      )
      .orderBy(desc(memories.createdAt))
      .limit(100);

    // Coordination sessions this period and previous period
    const [coordSessionsThis, coordSessionsPrev] = await Promise.all([
      db
        .select({
          id: coordSessions.id,
          startedAt: coordSessions.startedAt,
          status: coordSessions.status,
        })
        .from(coordSessions)
        .where(gte(coordSessions.startedAt, cutoff)),
      db
        .select({ id: coordSessions.id })
        .from(coordSessions)
        .where(
          and(
            gte(coordSessions.startedAt, prevCutoff),
            lt(coordSessions.startedAt, cutoff),
          ),
        ),
    ]);

    // Schedules with overdue detection
    const [schedulesList, recentRuns] = await Promise.all([
      db.select().from(schedules).orderBy(schedules.name),
      db
        .select({
          scheduleId: scheduleRuns.scheduleId,
          status: scheduleRuns.status,
          startedAt: scheduleRuns.startedAt,
        })
        .from(scheduleRuns)
        .orderBy(desc(scheduleRuns.startedAt))
        .limit(200),
    ]);

    // Build latest run status map
    const latestRunBySchedule = new Map<string, string>();
    for (const run of recentRuns) {
      if (run.scheduleId && !latestRunBySchedule.has(run.scheduleId)) {
        latestRunBySchedule.set(run.scheduleId, run.status ?? "unknown");
      }
    }

    const nowMs = Date.now();
    const schedulesOut = schedulesList.map((s: any) => {
      const effectiveStatus = latestRunBySchedule.get(s.id) ?? null;
      let overdue = false;
      if (s.enabled && s.lastRunAt) {
        const lastRun = new Date(s.lastRunAt).getTime();
        if (s.cronExpr) {
          // Simple overdue: more than 25h since last run for daily schedules
          overdue = (nowMs - lastRun) > 25 * 3600000;
        }
      }
      return {
        id: s.id,
        name: s.name,
        enabled: s.enabled,
        lastStatus: effectiveStatus,
        lastRunAt: s.lastRunAt,
        cronExpr: s.cronExpr,
        overdue,
      };
    });

    // Split period memories vs heatmap memories
    const periodMemories = allMemories.filter((m: any) => m.createdAt >= cutoff);

    // Aggregate period memories
    const byDay: Record<string, number> = {};
    const byType: Record<string, number> = {};
    const sessionIds = new Set<string>();
    let unscopedCount = 0;
    const priorities: number[] = [];

    // Per-project accumulators
    interface ProjectAcc {
      lastActive: string;
      totalMemories: number;
      sessionIds: Set<string>;
      distribution: Record<string, number>;
      activityByDay: Record<string, number>;
    }
    const projectMap = new Map<string, ProjectAcc>();

    for (const m of periodMemories) {
      const day = (m.createdAt as string).slice(0, 10);
      byDay[day] = (byDay[day] || 0) + 1;

      const type = (m.eventType as string) || "unknown";
      byType[type] = (byType[type] || 0) + 1;
      priorities.push((m.priority as number) ?? 3);

      // Parse metadata for session_id
      let meta: Record<string, unknown> | null = null;
      if (m.metadata) {
        try {
          meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata;
        } catch { /* ignore */ }
      }
      if (meta?.session_id) sessionIds.add(meta.session_id as string);

      // Project aggregation
      const rawProject = m.project as string | null;
      const displayName = rawProject || null;

      if (!displayName) {
        unscopedCount++;
        continue;
      }

      let acc = projectMap.get(displayName);
      if (!acc) {
        acc = {
          lastActive: m.createdAt as string,
          totalMemories: 0,
          sessionIds: new Set(),
          distribution: {},
          activityByDay: {},
        };
        projectMap.set(displayName, acc);
      }

      acc.totalMemories++;
      acc.distribution[type] = (acc.distribution[type] || 0) + 1;
      acc.activityByDay[day] = (acc.activityByDay[day] || 0) + 1;
      if ((m.createdAt as string) > acc.lastActive) acc.lastActive = m.createdAt as string;
      if (meta?.session_id) acc.sessionIds.add(meta.session_id as string);
    }

    // Build project cards
    const projectCards = Array.from(projectMap.entries())
      .map(([displayName, acc]) => {
        const daysSince = Math.floor(
          (now.getTime() - new Date(acc.lastActive).getTime()) / 86400000,
        );

        // Momentum from coordination sessions
        const sessThis = coordSessionsThis.length;
        const sessPrev = coordSessionsPrev.length || 1;
        let momentum: "up" | "steady" | "cooling" | "stalled";
        let momentumDelta = 0;
        if (sessPrev === 0 && sessThis === 0) {
          momentum = "stalled";
        } else if (sessPrev === 0) {
          momentum = "up";
          momentumDelta = 100;
        } else {
          momentumDelta = Math.round(((sessThis - sessPrev) / sessPrev) * 100);
          if (momentumDelta >= 25) momentum = "up";
          else if (momentumDelta >= -10) momentum = "steady";
          else if (momentumDelta >= -50) momentum = "cooling";
          else momentum = "stalled";
        }

        const alerts: string[] = [];
        if (daysSince > 3 && daysSince <= 14) alerts.push(`No sessions in ${daysSince} days`);
        if (daysSince > 14) alerts.push(`Inactive for ${daysSince} days`);

        const isActive = momentum !== "stalled" && daysSince <= 7;

        return {
          displayName,
          momentum,
          momentumDelta,
          sessionCount: acc.sessionIds.size,
          decisionCount: 0,
          taskCompletedCount: 0,
          narrative: null,
          lastActive: acc.lastActive,
          daysSinceLastSession: daysSince,
          isActive,
          alerts,
          recentDecisions: [],
          blockers: [],
          velocitySeries: [],
        };
      })
      .sort((a, b) => {
        if (a.isActive !== b.isActive) return a.isActive ? -1 : 1;
        const order = { up: 0, steady: 1, cooling: 2, stalled: 3 };
        return order[a.momentum] - order[b.momentum] || b.sessionCount - a.sessionCount;
      });

    // Build highlights by project
    const highlightsByProject = new Map<string, { content: string; type: string; createdAt: string }[]>();
    for (const h of highlights) {
      const displayName = h.project || null;
      if (!displayName) continue;
      let list = highlightsByProject.get(displayName);
      if (!list) {
        list = [];
        highlightsByProject.set(displayName, list);
      }
      if (list.length >= 10) continue;
      const raw = (h.content as string) || "";
      const cleaned = raw
        .replace(/^(Decision|Lesson|Lesson Learned):\s*/i, "")
        .replace(/\n/g, " ")
        .trim();
      const truncated = cleaned.length > 120 ? cleaned.slice(0, 117) + "..." : cleaned;
      if (truncated) {
        list.push({
          content: truncated,
          type: h.eventType as string,
          createdAt: h.createdAt as string,
        });
      }
    }

    // Heatmap
    const heatmap: Record<string, { total: number; types: Record<string, number>; projects: Record<string, number> }> = {};
    for (const row of allMemories) {
      const day = (row.createdAt as string).slice(0, 10);
      if (!heatmap[day]) heatmap[day] = { total: 0, types: {}, projects: {} };
      heatmap[day].total++;
      const type = (row.eventType as string) || "unknown";
      heatmap[day].types[type] = (heatmap[day].types[type] || 0) + 1;
      const proj = (row.project as string) || "Unscoped";
      heatmap[day].projects[proj] = (heatmap[day].projects[proj] || 0) + 1;
    }

    // Summary
    const totalMemories = periodMemories.length;
    const growthPct =
      prevCount === 0
        ? totalMemories > 0
          ? -1
          : 0
        : Math.round(((totalMemories - prevCount) / prevCount) * 100);

    // Allocation
    const totalCoordSessions = coordSessionsThis.length;
    const allocation: { displayName: string; percent: number; delta: number }[] = [];

    const memoryInsights = computeMemoryInsights({
      byType,
      total: totalMemories,
      activeSessions: sessionIds.size,
      growthPct,
      projects: Array.from(projectMap.entries()).map(([name, acc]) => ({
        displayName: name,
        totalMemories: acc.totalMemories,
      })),
      unscopedCount,
      priorities,
    });

    return new Response(
      JSON.stringify({
        period,
        summary: {
          totalMemories,
          growthPct,
          activeSessions: sessionIds.size,
          contentPublished: 0,
          publishSuccessRate: 100,
        },
        memory: {
          byDay,
          byType,
          total: totalMemories,
          totalAll,
          totalCloud: totalAll,
        },
        lastMemoryAt: allMemories[0]?.createdAt ?? null,
        content: null, // tweets/linkedin not available in self-hosted
        allocation,
        projects: projectCards,
        unscopedCount,
        schedules: schedulesOut,
        heatmap,
        memoryInsights,
        verification: undefined,
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
    console.error("[insights GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

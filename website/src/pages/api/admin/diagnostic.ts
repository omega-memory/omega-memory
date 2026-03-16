import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories, coordSessions } from "../../../lib/db/schema";
import { count, eq, gte, lte, gt, lt, and, or, isNull, sql } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/**
 * GET /api/admin/diagnostic?days=30
 *
 * System health diagnostic including memory health, quality metrics,
 * session counts, and recommendations.
 *
 * Note: Some production metrics (tool usage from coord_audit, LLM costs
 * from session_usage, latency) are not available in the self-hosted
 * Drizzle schema. Those sections return empty/zero gracefully.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const days = Math.min(
    Math.max(Number(url.searchParams.get("days")) || 30, 1),
    365,
  );
  const now = new Date();
  const cutoff = new Date(now.getTime() - days * 86_400_000).toISOString();
  const sevenDaysAgo = new Date(now.getTime() - 7 * 86_400_000).toISOString();
  const fourteenDaysAgo = new Date(now.getTime() - 14 * 86_400_000).toISOString();
  const oneMonthAgo = new Date(now.getTime() - 30 * 86_400_000).toISOString();
  const ninetyDaysAgo = new Date(now.getTime() - 90 * 86_400_000).toISOString();
  const sessionsCutoff = cutoff < ninetyDaysAgo ? cutoff : ninetyDaysAgo;

  try {
    // Run all queries in parallel
    const [
      memoryTotalRes,
      neverRes,
      lowRes,
      medRes,
      highRes,
      velocityRes,
      deadByTypeRes,
      sessionsRes,
      qualityRes,
      memoriesCreatedRes,
      memoriesAccessedRes,
    ] = await Promise.all([
      // Total memories
      db.select({ value: count() }).from(memories),
      // Never accessed
      db.select({ value: count() }).from(memories).where(
        or(eq(memories.accessCount, 0), isNull(memories.accessCount)),
      ),
      // Low access (1-2)
      db.select({ value: count() }).from(memories).where(
        and(gte(memories.accessCount, 1), lte(memories.accessCount, 2)),
      ),
      // Medium access (3-9)
      db.select({ value: count() }).from(memories).where(
        and(gte(memories.accessCount, 3), lte(memories.accessCount, 9)),
      ),
      // High access (10+)
      db.select({ value: count() }).from(memories).where(gte(memories.accessCount, 10)),
      // Velocity: memories created in last 7 days
      db
        .select({ eventType: memories.eventType })
        .from(memories)
        .where(gte(memories.createdAt, sevenDaysAgo))
        .limit(10000),
      // Dead memories: access_count=0, older than 14 days
      db
        .select({ eventType: memories.eventType })
        .from(memories)
        .where(and(eq(memories.accessCount, 0), lt(memories.createdAt, fourteenDaysAgo)))
        .limit(10000),
      // Sessions in window
      db
        .select({ startedAt: coordSessions.startedAt })
        .from(coordSessions)
        .where(gte(coordSessions.startedAt, sessionsCutoff)),
      // Quality metrics: priority, project
      db
        .select({
          priority: memories.priority,
          project: memories.project,
          extractedKeywords: memories.extractedKeywords,
        })
        .from(memories)
        .limit(10000),
      // Memories created in period
      db.select({ value: count() }).from(memories).where(gte(memories.createdAt, cutoff)),
      // Memories accessed in period
      db.select({ value: count() }).from(memories).where(gte(memories.lastAccessed, cutoff)),
    ]);

    // --- Memory Health ---
    const total = memoryTotalRes[0]?.value ?? 0;
    const never = neverRes[0]?.value ?? 0;
    const low = lowRes[0]?.value ?? 0;
    const med = medRes[0]?.value ?? 0;
    const high = highRes[0]?.value ?? 0;
    const accessed = total - never;
    const utilizationPct = total > 0 ? +((accessed / total) * 100).toFixed(1) : 0;

    // Velocity 7d
    const velocityMap: Record<string, number> = {};
    for (const row of velocityRes) {
      const t = row.eventType || "unknown";
      velocityMap[t] = (velocityMap[t] || 0) + 1;
    }
    const velocity7d = Object.entries(velocityMap)
      .map(([event_type, count]) => ({ event_type, count }))
      .sort((a, b) => b.count - a.count);
    const velocityTotal7d = velocityRes.length;

    // Dead by type
    const deadTypeMap: Record<string, number> = {};
    for (const row of deadByTypeRes) {
      const t = row.eventType || "unknown";
      deadTypeMap[t] = (deadTypeMap[t] || 0) + 1;
    }
    const deadByType = Object.entries(deadTypeMap)
      .map(([event_type, count]) => ({ event_type, count }))
      .sort((a, b) => b.count - a.count);
    const deadMemories = deadByTypeRes.length;
    const deadPct = total > 0 ? +((deadMemories / total) * 100).toFixed(1) : 0;

    // --- Quality Metrics ---
    const qualityRows = qualityRes;
    const priorities: number[] = qualityRows
      .map((r: typeof qualityRows[number]) => r.priority)
      .filter((p): p is number => p != null);
    const avgPriority =
      priorities.length > 0
        ? +(priorities.reduce((a: number, b: number) => a + b, 0) / priorities.length).toFixed(2)
        : 0;

    const priorityCountMap: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    for (const p of priorities) {
      if (p >= 1 && p <= 5) priorityCountMap[p]++;
    }
    const priorityDistribution = [1, 2, 3, 4, 5].map((priority) => ({
      priority,
      count: priorityCountMap[priority],
    }));

    const projectMap: Record<string, number> = {};
    let totalWithProject = 0;
    let totalWithTags = 0;
    for (const row of qualityRows) {
      if (row.project != null) {
        totalWithProject++;
        projectMap[row.project] = (projectMap[row.project] || 0) + 1;
      }
      if (row.extractedKeywords) {
        try {
          const tags = JSON.parse(row.extractedKeywords);
          if (Array.isArray(tags) && tags.length > 0) totalWithTags++;
        } catch { /* ignore */ }
      }
    }
    const byProject = Object.entries(projectMap)
      .map(([project, count]) => ({
        project: project.replace(/^\/Users\/[^/]+\/(Projects\/)?/, ""),
        count,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // --- Sessions ---
    const allSessions = sessionsRes;
    const totalSessions = allSessions.length;
    const weekSessions = allSessions.filter((s: typeof allSessions[number]) => s.startedAt && s.startedAt >= sevenDaysAgo).length;
    const monthSessions = allSessions.filter((s: typeof allSessions[number]) => s.startedAt && s.startedAt >= oneMonthAgo).length;
    const periodSessions = allSessions.filter((s: typeof allSessions[number]) => s.startedAt && s.startedAt >= cutoff).length;

    // --- OMEGA Usage (from memory activity) ---
    const memoriesCreated = memoriesCreatedRes[0]?.value ?? 0;
    const memoriesAccessed = memoriesAccessedRes[0]?.value ?? 0;
    const omegaCalls = memoriesCreated + memoriesAccessed;
    const omegaPerSession = periodSessions > 0 ? +(omegaCalls / periodSessions).toFixed(2) : 0;

    // --- Verdict ---
    const scaledThreshold = Math.round(50 * (days / 30));
    let verdict: "healthy" | "underused" | "idle";
    if (utilizationPct > 60 && omegaCalls > scaledThreshold) {
      verdict = "healthy";
    } else if (utilizationPct > 40 && omegaCalls >= Math.max(scaledThreshold * 0.1, 2)) {
      verdict = "underused";
    } else {
      verdict = "idle";
    }

    // --- Recommendations ---
    const recommendations: { severity: "critical" | "warning" | "info"; message: string; action?: string }[] = [];

    if (deadPct > 20)
      recommendations.push({
        severity: "critical",
        message: `${deadPct}% of memories are dead (never accessed, older than 14 days)`,
        action: "Review and purge stale memories",
      });
    else if (deadPct > 10)
      recommendations.push({
        severity: "warning",
        message: `${deadPct}% of memories are dead`,
        action: "Review memory creation patterns",
      });

    if (utilizationPct < 50)
      recommendations.push({
        severity: "warning",
        message: `Only ${utilizationPct}% of memories have ever been accessed`,
        action: "Check query patterns -- memories may not be matching",
      });

    if (omegaCalls === 0 && periodSessions > 10)
      recommendations.push({
        severity: "warning",
        message: "No OMEGA memory operations in any session this period",
        action: "Check MCP server connection and session start hook",
      });

    if (avgPriority < 2.5)
      recommendations.push({
        severity: "info",
        message: `Average memory priority is ${avgPriority.toFixed(1)}/5 -- most memories are low priority`,
        action: "Review priority assignment logic",
      });

    if (recommendations.length === 0)
      recommendations.push({
        severity: "info",
        message: "All systems operating normally",
        action: undefined,
      });

    const result = {
      memory_health: {
        total,
        utilization_pct: utilizationPct,
        velocity_7d: velocity7d,
        velocity_total_7d: velocityTotal7d,
        dead_memories: deadMemories,
        dead_pct: deadPct,
        dead_by_type: deadByType,
        access_buckets: { never, low, medium: med, high },
      },
      quality: {
        avg_priority: avgPriority,
        priority_distribution: priorityDistribution,
        by_project: byProject,
        total_with_tags: totalWithTags,
        total_with_project: totalWithProject,
        tags_pct: qualityRows.length > 0 ? +((totalWithTags / qualityRows.length) * 100).toFixed(0) : 0,
        project_pct: qualityRows.length > 0 ? +((totalWithProject / qualityRows.length) * 100).toFixed(0) : 0,
      },
      tool_usage: {
        top_tools: [],
        omega_tools: [],
        total_calls: 0,
        omega_calls: omegaCalls,
        omega_calls_audit: 0,
        omega_calls_memory_activity: omegaCalls,
        omega_per_session: omegaPerSession,
      },
      latency: {
        avg_ms: 0,
        p95_ms: 0,
        by_tool: [],
      },
      sessions: {
        total: totalSessions,
        week: weekSessions,
        month: monthSessions,
        period: periodSessions,
      },
      llm_costs: {
        total_usd: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_sessions: 0,
        by_model: [],
      },
      recommendations,
      verdict,
      period_days: days,
    };

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[diagnostic]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : "Unknown error" }),
      { status: 500, headers: JSON_HEADERS },
    );
  }
};

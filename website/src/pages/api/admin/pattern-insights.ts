import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { memories, coordSessions } from "../../../lib/db/schema";
import { eq, gte, lt, desc } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "private, max-age=300, stale-while-revalidate=600" };

const DIMENSION_LABELS: Record<string, string> = {
  git_workflow: "Git",
  session_timing: "Schedule",
  task_management: "Tasks",
  handoff_quality: "Handoffs",
  project_focus: "Projects",
  workflow_sequence: "Workflow",
  tool_preference: "Tools",
  co_edit_cluster: "Files",
};

const SKIP_PATTERN_TYPES = new Set([
  "memory_theme",
  "effectiveness_ranking",
  "topic_drift",
  "behavioral_drift",
  "knowledge_concentration",
]);

interface WorkflowMetricChange {
  label: string;
  thisWeek: number;
  lastWeek: number;
  direction: "up" | "down" | "flat";
  magnitude: number;
  unit: string;
  context: string;
}

interface WorkPattern {
  dimension: string;
  dimensionLabel: string;
  description: string;
  confidence: number;
  evidenceCount: number;
  sessionCount: number;
}

interface CoachingItem {
  id: string;
  text: string;
  category: string;
  impact: string;
  firstShown: string;
  trend: "improving" | "worsening" | "flat";
  graduated: boolean;
  graduatedAt?: string;
  evidence?: string;
}

function weekRange(weeksAgo: number): { start: string; end: string } {
  const now = new Date();
  const end = new Date(now);
  end.setDate(end.getDate() - weeksAgo * 7);
  const start = new Date(end);
  start.setDate(start.getDate() - 7);
  return { start: start.toISOString(), end: end.toISOString() };
}

function pctChange(current: number, previous: number): number {
  if (previous === 0) return current > 0 ? 100 : 0;
  return Math.round(((current - previous) / previous) * 100);
}

function direction(delta: number): "up" | "down" | "flat" {
  if (delta > 15) return "up";
  if (delta < -15) return "down";
  return "flat";
}

/**
 * GET /api/admin/pattern-insights
 *
 * Returns workflow changes (week-over-week metrics), behavioral
 * strength patterns, and coaching items derived from OMEGA memories.
 *
 * Some production metrics (handoffs, tasks, git commits) require tables
 * not in the self-hosted schema (coord_handoffs, coord_tasks, coord_git_events).
 * We compute what we can from coordSessions and memories.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const thisWeek = weekRange(0);
  const lastWeek = weekRange(1);

  try {
    const [
      sessionsThisWeek,
      sessionsLastWeek,
      patternsRes,
    ] = await Promise.all([
      db
        .select()
        .from(coordSessions)
        .where(gte(coordSessions.startedAt, thisWeek.start)),
      db
        .select()
        .from(coordSessions)
        .where(gte(coordSessions.startedAt, lastWeek.start)),
      db
        .select({
          content: memories.content,
          metadata: memories.metadata,
          createdAt: memories.createdAt,
        })
        .from(memories)
        .where(eq(memories.eventType, "behavioral_pattern"))
        .orderBy(desc(memories.createdAt))
        .limit(500),
    ]);

    // Filter sessions to the correct week windows in JS
    const sessThisWeek = sessionsThisWeek.filter(
      (s) => s.startedAt && s.startedAt >= thisWeek.start && s.startedAt < thisWeek.end,
    );
    const sessLastWeek = sessionsLastWeek.filter(
      (s) => s.startedAt && s.startedAt >= lastWeek.start && s.startedAt < lastWeek.end,
    );

    const changes: WorkflowMetricChange[] = [];
    const hasLastWeekBaseline = sessLastWeek.length > 0;

    // Session count change
    if (hasLastWeekBaseline) {
      const delta = pctChange(sessThisWeek.length, sessLastWeek.length);
      if (Math.abs(delta) > 15) {
        changes.push({
          label: "Sessions",
          thisWeek: sessThisWeek.length,
          lastWeek: sessLastWeek.length,
          direction: direction(delta),
          magnitude: Math.abs(delta),
          unit: "sessions",
          context: `${sessLastWeek.length} \u2192 ${sessThisWeek.length} sessions`,
        });
      }
    }

    // --- Parse behavioral patterns (strengths) ---
    const patterns = patternsRes;
    const seenKeys = new Set<string>();
    const bestByDimension = new Map<string, { content: string; patternType: string; confidence: number; evidenceCount: number; sessionCount: number }>();

    for (const row of patterns) {
      let meta: Record<string, unknown> = {};
      try { meta = row.metadata ? JSON.parse(row.metadata) : {}; } catch { continue; }

      const patternType = (meta.pattern_type as string) || "";
      const patternKey = (meta.pattern_key as string) || "";
      const confidence = (meta.confidence as number) || 0;
      const userConfirmed = meta.user_confirmed as boolean | null | undefined;
      if (userConfirmed === false) continue;
      if (confidence < 0.8) continue;
      if (SKIP_PATTERN_TYPES.has(patternType)) continue;
      if (seenKeys.has(patternKey)) continue;
      seenKeys.add(patternKey);

      const existing = bestByDimension.get(patternType);
      if (!existing || confidence > existing.confidence) {
        bestByDimension.set(patternType, {
          content: row.content || "",
          patternType,
          confidence,
          evidenceCount: (meta.evidence_count as number) || 0,
          sessionCount: (meta.evidence_sessions as number) || 0,
        });
      }
    }

    const strengths: WorkPattern[] = [];
    for (const [dimension, p] of bestByDimension) {
      strengths.push({
        dimension,
        dimensionLabel: DIMENSION_LABELS[dimension] || dimension,
        description: p.content,
        confidence: p.confidence,
        evidenceCount: p.evidenceCount,
        sessionCount: p.sessionCount,
      });
    }
    strengths.sort((a, b) => b.confidence - a.confidence);

    // --- Coaching items (simplified without handoff/task/git data) ---
    const coaching: CoachingItem[] = [];

    const totalSessionCount = sessThisWeek.length + sessLastWeek.length;

    const result = {
      changes,
      coaching: coaching.filter((c) => !c.graduated),
      strengths,
      hasData: changes.length > 0 || strengths.length > 0 || coaching.length > 0,
      sessionCount: totalSessionCount,
    };

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[pattern-insights]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Internal error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

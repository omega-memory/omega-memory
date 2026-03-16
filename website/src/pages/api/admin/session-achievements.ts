import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { coordSessions } from "../../../lib/db/schema";
import { eq, gte, lt, desc } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

interface SessionAchievement {
  session_id: string;
  project: string | null;
  task: string | null;
  started_at: string;
  duration_seconds: number | null;
  completed_tasks: string[];
  next_steps: string[];
  files_modified: string[];
  decisions_made: string[];
  blocked_items: string[];
  git_commits: { hash: string; message: string }[];
  has_unread_next_steps: boolean;
}

function parseTextArray(val: unknown): string[] {
  if (Array.isArray(val)) return val;
  if (typeof val === "string") {
    try {
      const parsed = JSON.parse(val);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return val.trim() ? [val] : [];
    }
  }
  return [];
}

/**
 * GET /api/admin/session-achievements?days=14&project=<slug>&cursor=<iso_date>
 *
 * Returns ended sessions as achievements. In production, this also merges
 * data from coord_handoffs. Since that table is not in the self-hosted
 * schema, we build achievements from coord_sessions metadata only.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const url = new URL(context.request.url);
  const days = Math.min(Number(url.searchParams.get("days")) || 14, 90);
  const project = url.searchParams.get("project") || null;
  const cursor = url.searchParams.get("cursor") || null;

  const since = new Date(Date.now() - days * 86_400_000).toISOString();

  try {
    // Fetch ended sessions within date range
    let sessions;
    if (cursor) {
      sessions = await db
        .select()
        .from(coordSessions)
        .where(eq(coordSessions.status, "ended"))
        .orderBy(desc(coordSessions.startedAt))
        .limit(50);

      // Filter in JS for gte(since) and lt(cursor)
      sessions = sessions.filter(
        (s) => s.startedAt && s.startedAt >= since && s.startedAt < cursor,
      );
    } else {
      sessions = await db
        .select()
        .from(coordSessions)
        .where(eq(coordSessions.status, "ended"))
        .orderBy(desc(coordSessions.startedAt))
        .limit(50);

      sessions = sessions.filter((s) => s.startedAt && s.startedAt >= since);
    }

    if (sessions.length === 0) {
      return new Response(JSON.stringify({ achievements: [], hasMore: false }), {
        status: 200,
        headers: JSON_HEADERS,
      });
    }

    // Build achievements from session data
    const achievements: SessionAchievement[] = sessions.map((s) => {
      let meta: Record<string, unknown> = {};
      try { meta = s.metadata ? JSON.parse(s.metadata) : {}; } catch { /* ignore */ }

      const durationSeconds = typeof meta.duration_seconds === "number"
        ? meta.duration_seconds
        : null;

      const gitCommits = Array.isArray(meta.git_commits)
        ? (meta.git_commits as { hash: string; message: string }[])
        : [];

      return {
        session_id: s.id,
        project: null,
        task: null,
        started_at: s.startedAt || "",
        duration_seconds: durationSeconds,
        completed_tasks: parseTextArray(meta.completed_tasks),
        next_steps: parseTextArray(meta.next_steps),
        files_modified: parseTextArray(meta.files_modified),
        decisions_made: parseTextArray(meta.decisions_made),
        blocked_items: parseTextArray(meta.blocked_items),
        git_commits: gitCommits,
        has_unread_next_steps: false,
      };
    });

    // Filter by project if requested
    const filtered = project
      ? achievements.filter((a) => a.project === project)
      : achievements;

    const hasMore = sessions.length === 50;
    const nextCursor = sessions.length > 0
      ? sessions[sessions.length - 1].startedAt
      : null;

    return new Response(
      JSON.stringify({ achievements: filtered, hasMore, nextCursor }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[session-achievements]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

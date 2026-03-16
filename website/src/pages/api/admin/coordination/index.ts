import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { coordSessions, coordFileClaims } from "../../../../lib/db/schema";
import { eq, desc, gte, lte, and, notInArray, inArray } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const MAX_AGE_HOURS = 12;

function isValidISODate(v: string): boolean {
  const d = new Date(v);
  return !isNaN(d.getTime());
}

/**
 * GET /api/admin/coordination
 *
 * Returns coordination sessions, file claims, and optionally all sub-resources.
 * Tables not yet in the Drizzle schema return empty arrays.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const params = new URL(context.request.url).searchParams;
    const since = params.get("since");
    const until = params.get("until");
    const includeAll = params.get("include") === "all";

    if (since && !isValidISODate(since)) {
      return new Response(JSON.stringify({ error: "Invalid 'since' date format" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }
    if (until && !isValidISODate(until)) {
      return new Response(JSON.stringify({ error: "Invalid 'until' date format" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    const isHistorical = !!(since || until);
    const cutoffStart = since || new Date(Date.now() - MAX_AGE_HOURS * 60 * 60_000).toISOString();
    const cutoffEnd = until || new Date().toISOString();

    // Build session query conditions
    const conditions = [];
    if (isHistorical) {
      conditions.push(gte(coordSessions.startedAt, cutoffStart));
      conditions.push(lte(coordSessions.startedAt, cutoffEnd));
    } else {
      // Use metadata or startedAt as proxy for heartbeat; exclude ended/stopped
      conditions.push(gte(coordSessions.startedAt, cutoffStart));
      conditions.push(notInArray(coordSessions.status, ["ended", "stopped"]));
    }

    const sessions = await db
      .select()
      .from(coordSessions)
      .where(and(...conditions))
      .orderBy(desc(coordSessions.startedAt))
      .limit(200);

    const sessionIds = sessions.map((s) => s.id);

    // Fetch file claims for these sessions
    const fileClaims = sessionIds.length > 0
      ? await db
          .select()
          .from(coordFileClaims)
          .where(inArray(coordFileClaims.sessionId, sessionIds))
          .orderBy(desc(coordFileClaims.claimedAt))
          .limit(500)
      : [];

    // Tables not in Drizzle schema — return empty arrays gracefully
    const basePayload: Record<string, unknown> = {
      sessions,
      file_claims: fileClaims,
      file_reads: [], // coord_file_reads not in schema
      is_historical: isHistorical,
    };

    if (includeAll) {
      basePayload.messages = [];
      basePayload.handoffs = [];
      basePayload.intents = [];
      basePayload.decisions = [];
      basePayload.tasks = [];
      basePayload.git_events = [];
      basePayload.metrics = [];
    }

    return new Response(JSON.stringify(basePayload), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[coordination]", err instanceof Error ? err.message : err);
    return new Response(
      JSON.stringify({ error: "Database error", sessions: [], file_claims: [], file_reads: [] }),
      { status: 500, headers: JSON_HEADERS },
    );
  }
};

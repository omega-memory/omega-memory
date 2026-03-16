import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories } from "../../../../lib/db/schema";
import { eq, and } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json" };

// Keep in sync with src/omega/types.py EventType enum
const KNOWN_EVENT_TYPES = new Set([
  "constraint", "checkpoint", "reminder", "decision", "lesson_learned",
  "error_pattern", "user_preference", "task_completion", "reflexion",
  "outcome_evaluation", "self_reflection", "session_summary",
  "preference_generated", "advisor_action_outcome", "advisor_insight",
  "sota_research", "research_report", "benchmark_update", "sota_scan",
  "file_conflict", "merge_claim", "merge_release", "file_claimed",
  "file_released", "branch_claimed", "branch_released", "session_respawn",
  "coordination_snapshot", "test", "code_chunk", "file_summary",
  "public_statement", "outcome_resolution", "contradiction_detected",
  "entity_profile_update",
]);

const EVENT_TYPE_RE = /^[a-zA-Z0-9_]{1,50}$/;
const NODE_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * PATCH /api/admin/memories/[id] - Update a memory's editable fields
 * Body: { content?: string, priority?: number, event_type?: string }
 *
 * The [id] parameter matches against node_id (UUID format).
 */
export const PATCH: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const id = context.params.id;
  if (!id || !NODE_ID_RE.test(id)) {
    return new Response(JSON.stringify({ error: "Invalid memory ID format" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  let body: Record<string, unknown>;
  try {
    body = await context.request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  const updates: Record<string, unknown> = {};

  // Validate content
  if (body.content !== undefined) {
    if (typeof body.content !== "string" || body.content.trim().length === 0) {
      return new Response(JSON.stringify({ error: "Content must be a non-empty string" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }
    if (body.content.length > 50_000) {
      return new Response(JSON.stringify({ error: "Content too long (max 50,000 characters)" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }
    updates.content = body.content.trim();
  }

  // Validate priority
  if (body.priority !== undefined) {
    const p = Number(body.priority);
    if (!Number.isInteger(p) || p < 1 || p > 5) {
      return new Response(JSON.stringify({ error: "Priority must be 1-5" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }
    updates.priority = p;
  }

  // Validate event_type
  if (body.event_type !== undefined) {
    const et = body.event_type as string;
    if (typeof body.event_type !== "string" || !EVENT_TYPE_RE.test(et)) {
      return new Response(
        JSON.stringify({ error: "event_type must be alphanumeric/underscores, max 50 chars" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }
    if (!KNOWN_EVENT_TYPES.has(et)) {
      console.warn(`[memories PATCH] Unknown event_type accepted: "${et}"`);
    }
    updates.eventType = et;
  }

  if (Object.keys(updates).length === 0) {
    return new Response(JSON.stringify({ error: "No updates provided" }), {
      status: 400,
      headers: JSON_HEADERS,
    });
  }

  try {
    // Build SET clause dynamically
    const setCols: Record<string, unknown> = {};
    if (updates.content !== undefined) setCols.content = updates.content as string;
    if (updates.priority !== undefined) setCols.priority = updates.priority as number;
    if (updates.eventType !== undefined) setCols.eventType = updates.eventType as string;

    const result = await db
      .update(memories)
      .set(setCols)
      .where(and(eq(memories.nodeId, id)))
      .returning();

    if (!result || result.length === 0) {
      return new Response(JSON.stringify({ error: "Memory not found" }), {
        status: 404,
        headers: JSON_HEADERS,
      });
    }

    return new Response(JSON.stringify({ memory: result[0] }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[memories PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/escalation?label=<optional_label>
 *
 * Escalation rules depend on external tables (escalation_rules) not present
 * in the self-hosted Drizzle schema. Returns graceful empty data.
 */
export const GET: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // escalation_rules table not in self-hosted schema — return empty
    return new Response(JSON.stringify({ rules: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[escalation GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

type ConditionType =
  | "consecutive_failures"
  | "approval_pending_hours"
  | "duration_exceeds_avg_pct"
  | "dead_letter";

type ActionType = "notify_dashboard" | "auto_disable" | "escalate_admin";

const VALID_CONDITION_TYPES: ConditionType[] = [
  "consecutive_failures",
  "approval_pending_hours",
  "duration_exceeds_avg_pct",
  "dead_letter",
];

const VALID_ACTION_TYPES: ActionType[] = [
  "notify_dashboard",
  "auto_disable",
  "escalate_admin",
];

/**
 * POST /api/admin/escalation
 *
 * Body: { action: "create" | "update" | "delete", ...data }
 * Stub: escalation_rules table not available in self-hosted schema.
 */
export const POST: APIRoute = async (context) => {
  const { error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const action = body?.action as string | undefined;

    if (!action || !["create", "update", "delete"].includes(action)) {
      return new Response(
        JSON.stringify({ error: "action must be 'create', 'update', or 'delete'" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    if (action === "delete") {
      const id = body.id as string | undefined;
      if (!id) {
        return new Response(JSON.stringify({ error: "id is required for delete" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    if (action === "create" || action === "update") {
      const scheduleLabel = body.scheduleLabel as string | undefined;
      const conditionType = body.conditionType as ConditionType | undefined;
      const actionType = body.actionType as ActionType | undefined;

      if (!scheduleLabel) {
        return new Response(JSON.stringify({ error: "scheduleLabel is required" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (!conditionType || !VALID_CONDITION_TYPES.includes(conditionType)) {
        return new Response(
          JSON.stringify({
            error: `conditionType must be one of: ${VALID_CONDITION_TYPES.join(", ")}`,
          }),
          { status: 400, headers: { "Content-Type": "application/json" } },
        );
      }
      if (!actionType || !VALID_ACTION_TYPES.includes(actionType)) {
        return new Response(
          JSON.stringify({
            error: `actionType must be one of: ${VALID_ACTION_TYPES.join(", ")}`,
          }),
          { status: 400, headers: { "Content-Type": "application/json" } },
        );
      }

      if (action === "update" && !body.id) {
        return new Response(JSON.stringify({ error: "id is required for update" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    return new Response(
      JSON.stringify({
        error: "Escalation rules not configured",
        message:
          "The escalation feature requires the escalation_rules table. " +
          "This table is not available in the self-hosted schema.",
      }),
      {
        status: 501,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err: unknown) {
    console.error("[escalation POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

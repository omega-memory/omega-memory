import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/approvals
 *
 * Returns all pending approval requests.
 * The production route uses getPendingApprovals/approveRequest/rejectRequest
 * helpers and a "job_approvals" table that don't exist in the Drizzle schema.
 * We return graceful empty data.
 *
 * POST /api/admin/approvals
 *   Body: { action: "approve" | "reject", approvalId: string, reason?: string }
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    // "job_approvals" table not in Drizzle schema — return empty list
    return new Response(JSON.stringify({ approvals: [] }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[approvals GET]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    let body: Record<string, unknown>;
    try {
      body = await context.request.json();
    } catch {
      return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    const { action, approvalId } = body as {
      action?: string;
      approvalId?: string;
    };

    if (!action || !approvalId) {
      return new Response(
        JSON.stringify({ error: "action and approvalId are required" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    if (action !== "approve" && action !== "reject") {
      return new Response(
        JSON.stringify({ error: "action must be 'approve' or 'reject'" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    // "job_approvals" table not in Drizzle schema
    return new Response(
      JSON.stringify({ error: "Approval not found" }),
      { status: 404, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[approvals POST]", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

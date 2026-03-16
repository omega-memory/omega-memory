import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { coordSessions } from "../../../lib/db/schema";
import { gte } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

interface VerificationLayer {
  label: string;
  verified: number;
  total: number;
  rho: number;
}

interface VerificationMetrics {
  overallRho: number;
  layers: VerificationLayer[];
}

/**
 * GET /api/admin/verification?period=30
 *
 * Verification Rate (rho) API:
 * rho = verified agent actions / total agent actions
 *
 * In the production environment this checks coord_handoffs, coord_tasks,
 * coord_decisions, and coord_external_actions tables. Since those tables
 * (except coord_sessions) are not in the self-hosted Drizzle schema,
 * we compute a simplified metric using sessions only.
 */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const url = new URL(context.request.url);
    const period = Math.min(
      90,
      Math.max(7, parseInt(url.searchParams.get("period") || "30", 10) || 30),
    );

    const cutoff = new Date(Date.now() - period * 86400000).toISOString();

    // Sessions with ended status = "verified" (agent completed work properly)
    const sessions = await db
      .select({
        id: coordSessions.id,
        status: coordSessions.status,
      })
      .from(coordSessions)
      .where(gte(coordSessions.startedAt, cutoff));

    const layers: VerificationLayer[] = [];

    // Layer 1: Sessions - ended sessions = verified (agent left a trail)
    if (sessions.length > 0) {
      const verifiedSessions = sessions.filter((s) => s.status === "ended").length;
      layers.push({
        label: "Sessions",
        verified: verifiedSessions,
        total: sessions.length,
        rho: Math.round((verifiedSessions / sessions.length) * 100),
      });
    }

    // Layers 2-4 (Tasks, Decisions, Actions) require tables not in self-hosted schema.
    // Return empty layers gracefully. The frontend handles missing layers.

    // Overall rho
    let overallRho = 0;
    if (layers.length > 0) {
      const totalVerified = layers.reduce((s, l) => s + l.verified, 0);
      const totalAll = layers.reduce((s, l) => s + l.total, 0);
      overallRho = totalAll > 0 ? Math.round((totalVerified / totalAll) * 100) : 0;
    }

    const verification: VerificationMetrics = { overallRho, layers };

    return new Response(JSON.stringify(verification), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch {
    // Return empty metrics rather than 500
    return new Response(
      JSON.stringify({ overallRho: 0, layers: [] } satisfies VerificationMetrics),
      { status: 200, headers: JSON_HEADERS },
    );
  }
};

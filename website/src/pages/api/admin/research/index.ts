import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../../lib/api/requireAuth";
import { memories } from "../../../../lib/db/schema";
import { eq, ne, desc, and, like, isNotNull, sql, count } from "drizzle-orm";

const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

/** GET /api/admin/research?limit=20&offset=0&subsystem=&source= */
export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const params = new URL(context.request.url).searchParams;
    const limit = parseInt(params.get("limit") || "20", 10);
    const offset = parseInt(params.get("offset") || "0", 10);
    const subsystem = params.get("subsystem");
    const source = params.get("source");

    // Build conditions for findings
    const conditions = [
      eq(memories.eventType, "research_report"),
      sql`json_extract(${memories.metadata}, '$.source') != 'sota_scanner_report'`,
    ];

    if (subsystem) {
      conditions.push(sql`json_extract(${memories.metadata}, '$.omega_subsystem') = ${subsystem}`);
    }
    if (source === "scanner") {
      conditions.push(sql`json_extract(${memories.metadata}, '$.source') = 'sota_scanner'`);
    } else if (source === "manual") {
      conditions.push(sql`json_extract(${memories.metadata}, '$.source') != 'sota_scanner'`);
    }

    // Findings query
    const findings = await db
      .select({
        id: memories.id,
        content: memories.content,
        metadata: memories.metadata,
        createdAt: memories.createdAt,
        priority: memories.priority,
      })
      .from(memories)
      .where(and(...conditions))
      .orderBy(desc(memories.createdAt))
      .limit(limit)
      .offset(offset);

    // Count query
    const [totalRow] = await db
      .select({ value: count() })
      .from(memories)
      .where(and(...conditions));
    const total = totalRow?.value ?? 0;

    const enriched = findings.map((f: typeof findings[number]) => {
      const meta = f.metadata ? (JSON.parse(f.metadata) as Record<string, unknown>) : {};
      const priorityScore = typeof meta.priority_score === "number" ? meta.priority_score : 0;
      return {
        id: f.id,
        content: f.content,
        created_at: f.createdAt,
        source_url: meta.source_url ?? null,
        source_type: meta.source_type ?? meta.tag ?? "unknown",
        technique: meta.technique ?? null,
        impact_score: typeof meta.impact_score === "number" ? meta.impact_score : null,
        feasibility_score: typeof meta.feasibility_score === "number" ? meta.feasibility_score : null,
        priority_score: priorityScore,
        subsystem: meta.omega_subsystem ?? null,
        source: meta.source ?? meta.tag ?? null,
        domain: meta.source_domain ?? null,
        curation_status: (meta.curation_status as string) ?? null,
      };
    });

    // Fetch plans
    const planRows = await db
      .select({
        id: memories.id,
        content: memories.content,
        metadata: memories.metadata,
        createdAt: memories.createdAt,
      })
      .from(memories)
      .where(eq(memories.eventType, "research_plan"))
      .orderBy(desc(memories.createdAt))
      .limit(100);

    const plans = planRows.map((p: typeof planRows[number]) => {
      const meta = p.metadata ? (JSON.parse(p.metadata) as Record<string, unknown>) : {};
      return {
        id: p.id,
        title: (meta.title as string) ?? "Untitled Plan",
        content: p.content,
        status: (meta.status as string) ?? "draft",
        subsystem: (meta.subsystem as string) ?? null,
        finding_id: (meta.finding_id as string) ?? null,
        created_at: p.createdAt,
      };
    });

    // Stats
    const [totalFindingsRow] = await db
      .select({ value: count() })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source') != 'sota_scanner_report'`,
        ),
      );

    const [scannerFindingsRow] = await db
      .select({ value: count() })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source') = 'sota_scanner'`,
        ),
      );

    const highPriorityRows = await db
      .select({ metadata: memories.metadata })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source') != 'sota_scanner_report'`,
          sql`json_extract(${memories.metadata}, '$.priority_score') IS NOT NULL`,
        ),
      );

    const highPriorityCount = highPriorityRows.filter((row: typeof highPriorityRows[number]) => {
      const meta = row.metadata ? (JSON.parse(row.metadata) as Record<string, unknown>) : {};
      return typeof meta.priority_score === "number" && meta.priority_score >= 12;
    }).length;

    const subsystemRows = await db
      .select({ metadata: memories.metadata })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source') != 'sota_scanner_report'`,
          sql`json_extract(${memories.metadata}, '$.omega_subsystem') IS NOT NULL`,
        ),
      );

    const subsystems = [
      ...new Set(
        subsystemRows
          .map((row: typeof subsystemRows[number]) => {
            const meta = row.metadata ? (JSON.parse(row.metadata) as Record<string, unknown>) : {};
            return meta.omega_subsystem as string | undefined;
          })
          .filter(Boolean),
      ),
    ];

    // Scan history
    const scanReports = await db
      .select({
        content: memories.content,
        metadata: memories.metadata,
        createdAt: memories.createdAt,
      })
      .from(memories)
      .where(
        and(
          eq(memories.eventType, "research_report"),
          sql`json_extract(${memories.metadata}, '$.source') = 'sota_scanner_report'`,
        ),
      )
      .orderBy(desc(memories.createdAt))
      .limit(10);

    const scanHistory = scanReports.map((r: typeof scanReports[number]) => {
      const meta = r.metadata ? (JSON.parse(r.metadata) as Record<string, unknown>) : {};
      return {
        scan_date: meta.scan_date ?? r.createdAt,
        raw_findings: meta.raw_findings ?? meta.findings_count ?? 0,
        kept: meta.kept ?? meta.high_priority_count ?? 0,
        discarded:
          typeof meta.discarded === "number"
            ? meta.discarded
            : typeof meta.raw_findings === "number" && typeof meta.kept === "number"
              ? (meta.raw_findings as number) - (meta.kept as number)
              : 0,
        duration_ms: meta.duration_ms ?? null,
        findings_summary: meta.findings_summary ?? [],
      };
    });

    return new Response(
      JSON.stringify({
        findings: enriched,
        plans,
        total,
        stats: {
          total_findings: totalFindingsRow?.value ?? 0,
          scanner_findings: scannerFindingsRow?.value ?? 0,
          high_priority: highPriorityCount,
          plan_count: plans.length,
          subsystems,
        },
        scan_history: scanHistory,
      }),
      { status: 200, headers: JSON_HEADERS },
    );
  } catch (err: unknown) {
    console.error("[research GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** POST /api/admin/research -- store a scan report or create/update a plan */
export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();

    // Plan creation/update
    if (body.type === "plan") {
      const { id, title, content, status, subsystem, finding_id } = body as {
        id?: number;
        title: string;
        content: string;
        status: string;
        subsystem: string | null;
        finding_id: string | null;
      };

      if (id) {
        await db
          .update(memories)
          .set({
            content,
            metadata: JSON.stringify({ title, status, subsystem, finding_id }),
            priority: 4,
          })
          .where(eq(memories.id, id));

        return new Response(JSON.stringify({ ok: true, id }), {
          status: 200,
          headers: JSON_HEADERS,
        });
      }

      // Create new plan
      const nodeId = `plan_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
      const result = await db
        .insert(memories)
        .values({
          nodeId,
          content,
          eventType: "research_plan",
          priority: 4,
          userId: user!.id,
          createdAt: new Date().toISOString(),
          metadata: JSON.stringify({ title, status, subsystem, finding_id }),
        })
        .returning({ id: memories.id });

      return new Response(JSON.stringify({ ok: true, id: result[0]?.id }), {
        status: 200,
        headers: JSON_HEADERS,
      });
    }

    // Scan report storage
    const {
      scan_date,
      bookmarks_scanned,
      papers_scanned,
      findings_count,
      high_priority_count,
      findings,
      trending_topics,
      report_path,
    } = body as {
      scan_date: string;
      bookmarks_scanned: number;
      papers_scanned: number;
      findings_count: number;
      high_priority_count: number;
      findings: Record<string, unknown>[];
      trending_topics: string[];
      report_path: string;
    };

    const nodeId = `scan_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    const result = await db
      .insert(memories)
      .values({
        nodeId,
        content: `SOTA scan ${scan_date}: ${bookmarks_scanned} bookmarks, ${papers_scanned} papers scanned. ${findings_count} findings, ${high_priority_count} high-priority. Trending: ${trending_topics.join(", ")}.`,
        eventType: "research_report",
        priority: 3,
        userId: user!.id,
        createdAt: new Date().toISOString(),
        metadata: JSON.stringify({
          source: "sota_scanner_report",
          scan_date,
          bookmarks_scanned,
          papers_scanned,
          findings_count,
          high_priority_count,
          trending_topics,
          report_path,
          findings_summary: findings.map((f) => ({
            technique: f.technique,
            priority_score: f.priority_score,
            subsystem: f.subsystem,
          })),
        }),
      })
      .returning({ id: memories.id });

    return new Response(JSON.stringify({ ok: true, id: result[0]?.id }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[research POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** PATCH /api/admin/research -- update curation status on a finding */
export const PATCH: APIRoute = async (context) => {
  const { db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const { id, curation_status } = (await context.request.json()) as {
      id: number;
      curation_status: "starred" | "dismissed" | "archived" | null;
    };

    if (!id) {
      return new Response(JSON.stringify({ error: "Missing id" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Fetch existing metadata to merge
    const [existing] = await db
      .select({ metadata: memories.metadata })
      .from(memories)
      .where(eq(memories.id, id))
      .limit(1);

    if (!existing) {
      return new Response(JSON.stringify({ error: "Not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    const meta = existing.metadata ? (JSON.parse(existing.metadata) as Record<string, unknown>) : {};
    const updatedMeta = { ...meta, curation_status };

    await db
      .update(memories)
      .set({ metadata: JSON.stringify(updatedMeta) })
      .where(eq(memories.id, id));

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[research PATCH]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

/** DELETE /api/admin/research?id=<id> -- delete a plan */
export const DELETE: APIRoute = async (context) => {
  const { db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const id = new URL(context.request.url).searchParams.get("id");
    if (!id) {
      return new Response(JSON.stringify({ error: "Missing id" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const numId = parseInt(id, 10);

    // Only allow deleting plans, not findings
    const [existing] = await db
      .select({ eventType: memories.eventType })
      .from(memories)
      .where(eq(memories.id, numId))
      .limit(1);

    if (!existing) {
      return new Response(JSON.stringify({ error: "Not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (existing.eventType !== "research_plan") {
      return new Response(JSON.stringify({ error: "Can only delete plans" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await db.delete(memories).where(eq(memories.id, numId));

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[research DELETE]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

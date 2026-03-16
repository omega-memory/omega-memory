import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db";
import { memories } from "../../../lib/db/schema";
import { AuthEngine, getSessionFromCookie } from "../../../lib/auth";
import { count, desc, eq, like, and } from "drizzle-orm";

async function requireAuth(context: Parameters<APIRoute>[0]) {
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);
  if (!token) return null;

  const db = getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "dev-secret";
  const auth = new AuthEngine(db, secret);
  return auth.validateSession(token);
}

export const GET: APIRoute = async (context) => {
  const user = await requireAuth(context);
  if (!user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const db = getDb(context);
  const url = context.url;
  const page = Math.max(1, parseInt(url.searchParams.get("page") ?? "1"));
  const perPage = Math.min(100, Math.max(1, parseInt(url.searchParams.get("per_page") ?? "50")));
  const search = url.searchParams.get("q") ?? "";
  const eventType = url.searchParams.get("event_type") ?? "";
  const project = url.searchParams.get("project") ?? "";

  const conditions = [];
  if (search) conditions.push(like(memories.content, `%${search}%`));
  if (eventType) conditions.push(eq(memories.eventType, eventType));
  if (project) conditions.push(eq(memories.project, project));

  const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

  const [totalRow] = await db.select({ value: count() }).from(memories).where(whereClause);
  const rows = await db
    .select()
    .from(memories)
    .where(whereClause)
    .orderBy(desc(memories.createdAt))
    .limit(perPage)
    .offset((page - 1) * perPage);

  return new Response(
    JSON.stringify({
      data: rows,
      total: totalRow.value,
      page,
      perPage,
      totalPages: Math.max(1, Math.ceil(totalRow.value / perPage)),
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }
  );
};

export const PATCH: APIRoute = async (context) => {
  const user = await requireAuth(context);
  if (!user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const db = getDb(context);
  const body = await context.request.json();
  const { id, ...updates } = body;

  if (!id) {
    return new Response(JSON.stringify({ error: "Memory id is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Only allow safe fields to be updated
  const allowedFields = new Set([
    "content",
    "status",
    "priority",
    "eventType",
    "project",
    "metadata",
    "ttlSeconds",
  ]);

  const safeUpdates: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(updates)) {
    if (allowedFields.has(key)) {
      safeUpdates[key] = value;
    }
  }

  if (Object.keys(safeUpdates).length === 0) {
    return new Response(JSON.stringify({ error: "No valid fields to update" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  await db.update(memories).set(safeUpdates).where(eq(memories.id, id));

  const [updated] = await db.select().from(memories).where(eq(memories.id, id)).limit(1);

  return new Response(JSON.stringify({ ok: true, data: updated ?? null }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

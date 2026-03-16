import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { users } from "../../../lib/db/schema";
import { eq, asc } from "drizzle-orm";

/**
 * GET  /api/admin/users — list all users (admin only)
 * POST /api/admin/users — add a user
 * DELETE /api/admin/users — remove a user (body: { id })
 *
 * The production route queries an "allowed_users" table. We use the "users"
 * table from the Drizzle schema as the equivalent.
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const rows = await db
      .select({
        id: users.id,
        email: users.email,
        role: users.role,
        createdAt: users.createdAt,
      })
      .from(users)
      .orderBy(asc(users.createdAt));

    return new Response(JSON.stringify(rows), {
      status: 200,
      headers: { ...JSON_HEADERS, "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    console.error("[admin/users GET]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const POST: APIRoute = async (context) => {
  const { user: authUser, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const email: string | undefined = body.email;
    const userRole: string = body.userRole ?? "user";

    if (!email || !email.includes("@")) {
      return new Response(
        JSON.stringify({ error: "A valid email is required" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    const id = crypto.randomUUID();
    const now = new Date().toISOString();

    try {
      await db.insert(users).values({
        id,
        email: email.toLowerCase().trim(),
        role: userRole,
        createdAt: now,
        updatedAt: now,
      });
    } catch (insertErr: unknown) {
      // SQLite UNIQUE constraint violation
      const msg = insertErr instanceof Error ? insertErr.message : "";
      if (msg.includes("UNIQUE") || msg.includes("unique")) {
        return new Response(
          JSON.stringify({ error: "User already exists" }),
          { status: 409, headers: JSON_HEADERS },
        );
      }
      throw insertErr;
    }

    const [created] = await db
      .select({
        id: users.id,
        email: users.email,
        role: users.role,
        createdAt: users.createdAt,
      })
      .from(users)
      .where(eq(users.id, id))
      .limit(1);

    return new Response(JSON.stringify(created), {
      status: 201,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[admin/users POST]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

export const DELETE: APIRoute = async (context) => {
  const { user: authUser, db, error } = await requireAdminAuth(context);
  if (error) return error;

  try {
    const body = await context.request.json();
    const id: string | undefined = body.id;

    if (!id) {
      return new Response(
        JSON.stringify({ error: "id is required" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    // Prevent deleting an admin/superadmin (self-deletion guard)
    const [target] = await db
      .select({ role: users.role })
      .from(users)
      .where(eq(users.id, id))
      .limit(1);

    if (target?.role === "admin" || target?.role === "superadmin") {
      return new Response(
        JSON.stringify({ error: "Cannot remove an admin" }),
        { status: 400, headers: JSON_HEADERS },
      );
    }

    await db.delete(users).where(eq(users.id, id));

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  } catch (err: unknown) {
    console.error("[admin/users DELETE]", err instanceof Error ? err.message : err);
    return new Response(JSON.stringify({ error: "Database error" }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
};

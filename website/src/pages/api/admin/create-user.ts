import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db/index";
import { AuthEngine } from "../../../lib/auth/index";
import { users } from "../../../lib/db/schema";
import { count } from "drizzle-orm";

/**
 * POST /api/admin/create-user — Create an admin user.
 *
 * Security: This endpoint only works when:
 *   1. No users exist yet (initial setup), OR
 *   2. The request includes a valid SETUP_SECRET header, OR
 *   3. The request comes from an authenticated admin session.
 *
 * Body: { email, password, displayName? }
 */
export const POST: APIRoute = async (context) => {
  try {
    const db = await getDb(context);

    // Check if any users exist
    const [userCount] = await db.select({ value: count() }).from(users);
    const hasUsers = (userCount?.value ?? 0) > 0;

    if (hasUsers) {
      // If users exist, require either SETUP_SECRET or authenticated session
      const setupSecret = context.request.headers.get("x-setup-secret");
      const expectedSecret =
        (context.locals.runtime as any)?.env?.SETUP_SECRET ??
        process.env.SETUP_SECRET;

      if (!expectedSecret || setupSecret !== expectedSecret) {
        // Try session auth as fallback
        const { getSessionFromCookie } = await import("../../../lib/auth/index");
        const cookie = context.request.headers.get("cookie");
        const token = getSessionFromCookie(cookie);

        if (!token) {
          return new Response(
            JSON.stringify({
              error: "Forbidden",
              detail: "Users already exist. Provide X-Setup-Secret header or authenticate first.",
            }),
            { status: 403, headers: { "Content-Type": "application/json" } },
          );
        }

        const secret =
          (context.locals.runtime as any)?.env?.AUTH_SECRET ??
          process.env.AUTH_SECRET ??
          "omega-dev-secret-change-in-production";
        const auth = new AuthEngine(db, secret);
        const sessionUser = await auth.validateSession(token);

        if (!sessionUser) {
          return new Response(
            JSON.stringify({ error: "Unauthorized" }),
            { status: 401, headers: { "Content-Type": "application/json" } },
          );
        }
      }
    }

    // Parse request body
    const body = await context.request.json();
    const { email, password, displayName } = body;

    if (!email || !password) {
      return new Response(
        JSON.stringify({ error: "email and password are required" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    if (password.length < 8) {
      return new Response(
        JSON.stringify({ error: "Password must be at least 8 characters" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    // Create user
    const secret =
      (context.locals.runtime as any)?.env?.AUTH_SECRET ??
      process.env.AUTH_SECRET ??
      "omega-dev-secret-change-in-production";
    const auth = new AuthEngine(db, secret);
    const newUser = await auth.createUser(email, password, displayName);

    return new Response(
      JSON.stringify({
        success: true,
        user: {
          id: newUser.id,
          email: newUser.email,
          displayName: newUser.displayName,
        },
      }),
      { status: 201, headers: { "Content-Type": "application/json" } },
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);

    // Handle duplicate email
    if (message.includes("UNIQUE") || message.includes("unique") || message.includes("duplicate")) {
      return new Response(
        JSON.stringify({ error: "A user with this email already exists" }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      );
    }

    console.error("[create-user POST]", message);
    return new Response(
      JSON.stringify({ error: "Failed to create user" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
};

import { getSessionFromCookie } from "../auth/index";
import { AuthEngine } from "../auth/index";
import { getDb } from "../db/index";
import type { APIContext } from "astro";

const JSON_HEADERS = { "Content-Type": "application/json" };

/**
 * Authenticate an admin request from cookie session.
 * Returns { user, db, error } — check error first.
 */
export async function requireAdminAuth(context: APIContext) {
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);
  if (!token) {
    return {
      user: null,
      db: null,
      error: new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: JSON_HEADERS,
      }),
    };
  }

  const db = await getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "omega-dev-secret-change-in-production";
  const auth = new AuthEngine(db, secret);
  const user = await auth.validateSession(token);

  if (!user) {
    return {
      user: null,
      db: null,
      error: new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: JSON_HEADERS,
      }),
    };
  }

  return { user, db, error: null };
}

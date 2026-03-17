import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db";
import { AuthEngine, getSessionFromCookie } from "../../../lib/auth";
import { webauthnCredentials } from "../../../lib/db/schema";
import { eq } from "drizzle-orm";

export const GET: APIRoute = async (context) => {
  const db = await getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "omega-dev-secret-change-in-production";

  // Require auth session
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);
  if (!token) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const auth = new AuthEngine(db, secret);
  const user = await auth.validateSession(token);
  if (!user) {
    return new Response(JSON.stringify({ error: "Invalid session" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const creds = await db
    .select({
      id: webauthnCredentials.id,
      credentialId: webauthnCredentials.credentialId,
      deviceType: webauthnCredentials.deviceType,
      backedUp: webauthnCredentials.backedUp,
      createdAt: webauthnCredentials.createdAt,
      lastUsedAt: webauthnCredentials.lastUsedAt,
    })
    .from(webauthnCredentials)
    .where(eq(webauthnCredentials.userId, user.id));

  return new Response(
    JSON.stringify({
      hasPasskey: creds.length > 0,
      credentials: creds,
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
};

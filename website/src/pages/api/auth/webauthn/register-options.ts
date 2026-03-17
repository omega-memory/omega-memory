import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { AuthEngine, getSessionFromCookie } from "../../../../lib/auth";
import { users } from "../../../../lib/db/schema";
import { eq } from "drizzle-orm";
import { generateWebAuthnRegistrationOptions } from "../../../../lib/auth/providers/webauthn";

export const POST: APIRoute = async (context) => {
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

  try {
    const options = await generateWebAuthnRegistrationOptions(
      db,
      user.id,
      user.email,
      user.displayName ?? undefined,
    );

    return new Response(JSON.stringify(options), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

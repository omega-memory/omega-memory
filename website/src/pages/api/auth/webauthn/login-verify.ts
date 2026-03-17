import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { setSessionCookie } from "../../../../lib/auth";
import { verifyWebAuthnAuthentication } from "../../../../lib/auth/providers/webauthn";

export const POST: APIRoute = async (context) => {
  const db = await getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "omega-dev-secret-change-in-production";

  try {
    const body = await context.request.json();
    const { credentialResponse, challenge } = body;

    if (!credentialResponse || !challenge) {
      return new Response(JSON.stringify({ error: "Missing credentialResponse or challenge" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const result = await verifyWebAuthnAuthentication(
      db,
      secret,
      credentialResponse,
      challenge,
      context.request,
    );

    if (!result) {
      return new Response(JSON.stringify({ error: "Authentication failed" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }

    const cookie = setSessionCookie(result.sessionToken);

    return new Response(
      JSON.stringify({
        ok: true,
        user: {
          id: result.user.id,
          email: result.user.email,
          displayName: result.user.displayName,
        },
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Set-Cookie": cookie,
        },
      },
    );
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
};

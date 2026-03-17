import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { setSessionCookie } from "../../../../lib/auth";
import { OAuthProvider, getEnv } from "../../../../lib/auth/providers/oauth";

export const GET: APIRoute = async (context) => {
  const url = new URL(context.request.url);
  const error = url.searchParams.get("error");

  if (error) {
    const desc = url.searchParams.get("error_description") ?? error;
    return context.redirect(`/admin/login?error=${encodeURIComponent(desc)}`);
  }

  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");

  if (!code || !state) {
    return context.redirect("/admin/login?error=missing_params");
  }

  const db = await getDb(context);
  const env = getEnv(context.locals as Record<string, unknown>);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "omega-dev-secret-change-in-production";

  const oauth = new OAuthProvider();
  oauth.setEnv(env);

  try {
    const result = await oauth.handleCallback(context.request, db, secret);

    if (!result) {
      return context.redirect("/admin/login?error=oauth_failed");
    }

    const cookie = setSessionCookie(result.sessionToken);

    return new Response(null, {
      status: 302,
      headers: {
        Location: "/admin",
        "Set-Cookie": cookie,
      },
    });
  } catch (err) {
    console.error("OAuth callback error:", err);
    return context.redirect("/admin/login?error=oauth_error");
  }
};

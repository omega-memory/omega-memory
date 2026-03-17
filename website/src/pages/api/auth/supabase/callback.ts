/**
 * Supabase Auth callback endpoint.
 *
 * GET  ?access_token=...  — server-side callback
 * POST { access_token }   — client-side callback (from hash fragment extraction)
 *
 * Validates the Supabase JWT, upserts the user, creates a session, sets the cookie,
 * and redirects to /admin.
 */
import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { setSessionCookie } from "../../../../lib/auth";
import { SupabaseAuthProvider } from "../../../../lib/auth/providers/supabase";

function getEnv(name: string, context: any): string {
  const runtime = context.locals?.runtime as { env?: Record<string, string> } | undefined;
  return runtime?.env?.[name] ?? process.env[name] ?? "";
}

const supabase = new SupabaseAuthProvider();

export const GET: APIRoute = async (context) => {
  const db = await getDb(context);
  const authSecret = getEnv("AUTH_SECRET", context) || "omega-dev-secret-change-in-production";
  const jwtSecret = getEnv("SUPABASE_JWT_SECRET", context);

  if (!jwtSecret) {
    return new Response(JSON.stringify({ error: "SUPABASE_JWT_SECRET not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Extract access_token from query params
  const accessToken = context.url.searchParams.get("access_token");
  if (!accessToken) {
    return new Response(JSON.stringify({ error: "Missing access_token" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Create a synthetic request with the token for the provider
  const syntheticRequest = new Request(context.request.url, {
    headers: {
      "Authorization": `Bearer ${accessToken}`,
      "User-Agent": context.request.headers.get("user-agent") || "",
    },
  });

  const result = await supabase.authenticateWithEnv(syntheticRequest, db, authSecret, jwtSecret);
  if (!result) {
    return context.redirect("/admin/login?error=auth_failed");
  }

  const cookie = setSessionCookie(result.sessionToken);
  return new Response(null, {
    status: 302,
    headers: {
      Location: "/admin",
      "Set-Cookie": cookie,
    },
  });
};

export const POST: APIRoute = async (context) => {
  const db = await getDb(context);
  const authSecret = getEnv("AUTH_SECRET", context) || "omega-dev-secret-change-in-production";
  const jwtSecret = getEnv("SUPABASE_JWT_SECRET", context);

  if (!jwtSecret) {
    return new Response(JSON.stringify({ error: "SUPABASE_JWT_SECRET not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Parse JSON body
  let accessToken: string | null = null;
  try {
    const body = await context.request.json();
    accessToken = body.access_token ?? null;
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!accessToken) {
    return new Response(JSON.stringify({ error: "Missing access_token" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Create a synthetic request with the token
  const syntheticRequest = new Request(context.request.url, {
    headers: {
      "Authorization": `Bearer ${accessToken}`,
      "User-Agent": context.request.headers.get("user-agent") || "",
    },
  });

  const result = await supabase.authenticateWithEnv(syntheticRequest, db, authSecret, jwtSecret);
  if (!result) {
    return new Response(JSON.stringify({ error: "Invalid or expired token" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const cookie = setSessionCookie(result.sessionToken);
  return new Response(JSON.stringify({ ok: true, user: { id: result.user.id, email: result.user.email } }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": cookie,
    },
  });
};

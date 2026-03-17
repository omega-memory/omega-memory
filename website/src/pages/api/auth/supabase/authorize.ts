/**
 * Supabase OAuth authorization redirect.
 *
 * GET /api/auth/supabase/authorize?provider=google
 *
 * Redirects to Supabase Auth which handles the entire OAuth flow.
 * After auth, Supabase redirects back to /auth/supabase-callback with the JWT.
 */
import type { APIRoute } from "astro";
import { SupabaseAuthProvider } from "../../../../lib/auth/providers/supabase";

function getEnv(name: string, context: any): string {
  const runtime = context.locals?.runtime as { env?: Record<string, string> } | undefined;
  return runtime?.env?.[name] ?? process.env[name] ?? "";
}

const supabase = new SupabaseAuthProvider();

export const GET: APIRoute = async (context) => {
  const provider = context.url.searchParams.get("provider");
  if (!provider) {
    return new Response(JSON.stringify({ error: "Missing ?provider= query parameter" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const supabaseUrl = getEnv("SUPABASE_URL", context);
  if (!supabaseUrl) {
    return new Response(JSON.stringify({ error: "SUPABASE_URL not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Build the redirect URI — Supabase will send the user back here after auth
  const origin = context.url.origin;
  const redirectUri = `${origin}/auth/supabase-callback`;

  const authUrl = supabase.getAuthorizationUrlWithEnv(provider, redirectUri, supabaseUrl);

  return new Response(null, {
    status: 302,
    headers: { Location: authUrl },
  });
};

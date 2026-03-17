import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { OAuthProvider, getEnv, getEnabledProviders, getClientCredentials } from "../../../../lib/auth/providers/oauth";

export const GET: APIRoute = async (context) => {
  const url = new URL(context.request.url);
  const provider = url.searchParams.get("provider");

  if (!provider) {
    return new Response(JSON.stringify({ error: "Missing ?provider= parameter" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const env = getEnv(context.locals as Record<string, unknown>);
  const enabled = getEnabledProviders(env);

  if (!enabled.includes(provider.toLowerCase())) {
    return new Response(JSON.stringify({ error: `Provider "${provider}" is not enabled` }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const creds = getClientCredentials(env, provider);
  if (!creds) {
    return new Response(JSON.stringify({ error: `Missing credentials for ${provider}` }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const db = await getDb(context);

  // Build the callback URL (same origin)
  const callbackUri = `${url.origin}/api/auth/oauth/callback`;

  const oauth = new OAuthProvider();
  const authorizationUrl = await oauth.authorize(
    provider.toLowerCase(),
    callbackUri,
    db,
    env,
  );

  return new Response(null, {
    status: 302,
    headers: { Location: authorizationUrl },
  });
};

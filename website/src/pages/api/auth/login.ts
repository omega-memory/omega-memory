import type { APIRoute } from "astro";
import { getAuthProvider } from "../../../lib/auth/factory";
import { setSessionCookie } from "../../../lib/auth/index";
import { getDb } from "../../../lib/db";

export const POST: APIRoute = async (context) => {
  const db = await getDb(context);
  const env = (context.locals.runtime as any)?.env;
  const secret = env?.AUTH_SECRET ?? process.env.AUTH_SECRET ?? "omega-dev-secret-change-in-production";
  const provider = getAuthProvider();

  const contentType = context.request.headers.get("content-type") ?? "";

  // Parse "next" redirect target before passing request to provider
  let next = "/admin";
  if (contentType.includes("application/json")) {
    // We need to clone the request so the provider can also read the body
    const cloned = context.request.clone();
    try {
      const body = await cloned.json();
      next = body.next ?? next;
    } catch {
      // ignore parse errors, provider will handle validation
    }
  } else {
    const cloned = context.request.clone();
    try {
      const form = await cloned.formData();
      next = (form.get("next") as string) ?? next;
    } catch {
      // ignore parse errors
    }
  }

  const result = await provider.authenticate(context.request, db, secret);

  if (!result) {
    if (!contentType.includes("application/json")) {
      return context.redirect(`/admin/login?error=invalid&next=${encodeURIComponent(next)}`);
    }
    return new Response(JSON.stringify({ error: "Invalid credentials" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const cookie = setSessionCookie(result.sessionToken);

  if (!contentType.includes("application/json")) {
    return new Response(null, {
      status: 302,
      headers: {
        Location: next,
        "Set-Cookie": cookie,
      },
    });
  }

  return new Response(JSON.stringify({ ok: true, user: result.user }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": cookie,
    },
  });
};

import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db";
import { AuthEngine, setSessionCookie } from "../../../lib/auth";

export const POST: APIRoute = async (context) => {
  const db = await getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "omega-dev-secret-change-in-production";
  const auth = new AuthEngine(db, secret);

  let email: string;
  let password: string;
  let next = "/admin";

  // Parse body — support both form data and JSON
  const contentType = context.request.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const body = await context.request.json();
    email = body.email;
    password = body.password;
    next = body.next ?? next;
  } else {
    const form = await context.request.formData();
    email = form.get("email") as string;
    password = form.get("password") as string;
    next = (form.get("next") as string) ?? next;
  }

  if (!email || !password) {
    // Form submission — redirect back with error
    if (!contentType.includes("application/json")) {
      return context.redirect(`/admin/login?error=invalid&next=${encodeURIComponent(next)}`);
    }
    return new Response(JSON.stringify({ error: "Email and password are required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const user = await auth.authenticateUser(email, password);
  if (!user) {
    if (!contentType.includes("application/json")) {
      return context.redirect(`/admin/login?error=invalid&next=${encodeURIComponent(next)}`);
    }
    return new Response(JSON.stringify({ error: "Invalid credentials" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const deviceInfo = context.request.headers.get("user-agent") ?? undefined;
  const token = await auth.createSession(user.id, deviceInfo);
  const cookie = setSessionCookie(token);

  if (!contentType.includes("application/json")) {
    return new Response(null, {
      status: 302,
      headers: {
        Location: next,
        "Set-Cookie": cookie,
      },
    });
  }

  return new Response(JSON.stringify({ ok: true, user: { id: user.id, email: user.email } }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": cookie,
    },
  });
};

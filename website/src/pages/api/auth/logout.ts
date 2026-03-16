import type { APIRoute } from "astro";
import { getDb } from "../../../lib/db";
import {
  AuthEngine,
  clearSessionCookie,
  getSessionFromCookie,
} from "../../../lib/auth";

export const POST: APIRoute = async (context) => {
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);

  if (token) {
    try {
      const db = getDb(context);
      const secret =
        (context.locals.runtime as any)?.env?.AUTH_SECRET ??
        process.env.AUTH_SECRET ??
        "dev-secret";
      const auth = new AuthEngine(db, secret);
      await auth.revokeSession(token);
    } catch {
      // Best effort — clear cookie regardless
    }
  }

  const contentType = context.request.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Set-Cookie": clearSessionCookie(),
      },
    });
  }

  return new Response(null, {
    status: 302,
    headers: {
      Location: "/admin/login",
      "Set-Cookie": clearSessionCookie(),
    },
  });
};

// Also support GET for simple logout links
export const GET: APIRoute = async (context) => {
  const cookie = context.request.headers.get("cookie");
  const token = getSessionFromCookie(cookie);

  if (token) {
    try {
      const db = getDb(context);
      const secret =
        (context.locals.runtime as any)?.env?.AUTH_SECRET ??
        process.env.AUTH_SECRET ??
        "dev-secret";
      const auth = new AuthEngine(db, secret);
      await auth.revokeSession(token);
    } catch {
      // Best effort
    }
  }

  return new Response(null, {
    status: 302,
    headers: {
      Location: "/admin/login",
      "Set-Cookie": clearSessionCookie(),
    },
  });
};

/**
 * Astro middleware — protects /admin/* routes.
 * Works on both Cloudflare Workers and Node.js.
 */
import { defineMiddleware } from "astro:middleware";
import { getSessionFromCookie, getProSessionFromCookie } from "./lib/auth";

const PUBLIC_PATHS = new Set([
  "/admin/login",
  "/admin/onboarding",
  "/health",
  "/api/auth/login",
  "/api/auth/logout",
  "/api/auth/supabase/callback",
  "/api/auth/supabase/authorize",
  "/auth/supabase-callback",
  "/api/auth/oauth/authorize",
  "/api/auth/oauth/callback",
  "/api/auth/provider-config",
  "/api/auth/webauthn/login-options",
  "/api/auth/webauthn/login-verify",
  "/api/pro/auth",
  "/api/pro/logout",
  "/api/activate",
  "/api/webhook",
]);

export const onRequest = defineMiddleware(async (context, next) => {
  const { pathname } = context.url;

  // Public paths — always allowed
  if (PUBLIC_PATHS.has(pathname)) {
    return next();
  }

  // Admin routes require auth
  if (pathname.startsWith("/admin")) {
    const cookie = context.request.headers.get("cookie");
    const token = getSessionFromCookie(cookie);
    if (!token) {
      return context.redirect(`/admin/login?next=${encodeURIComponent(pathname)}`);
    }
    // Full validation happens in the Admin layout
  }

  // Pro routes require pro session
  if (pathname.startsWith("/pro/dashboard")) {
    const cookie = context.request.headers.get("cookie");
    const proToken = getProSessionFromCookie(cookie);
    if (!proToken) {
      return context.redirect("/pro");
    }
  }

  // API routes handle their own auth
  return next();
});

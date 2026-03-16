import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";

/**
 * GET /api/admin/me — Return the current authenticated user.
 *
 * Self-hosted: all authenticated users are treated as "owner".
 */
export const GET: APIRoute = async (context) => {
  const { user, error } = await requireAdminAuth(context);
  if (error) return error;

  return new Response(
    JSON.stringify({
      id: user.id,
      email: user.email,
      name: user.displayName ?? null,
      avatar: null,
      role: "owner", // self-hosted: all users are owners
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
};

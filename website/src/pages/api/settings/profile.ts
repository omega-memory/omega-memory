import type { APIRoute } from "astro";
import { requireAdminAuth } from "../../../lib/api/requireAuth";
import { users } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export const GET: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  return Response.json({
    name: user!.displayName ?? "",
    timezone: "UTC",
    avatar_url: null,
    preferences: {},
  });
};

export const POST: APIRoute = async (context) => {
  const { user, db, error } = await requireAdminAuth(context);
  if (error) return error;

  const body = await context.request.json();

  await db!
    .update(users)
    .set({
      displayName: body.name ?? undefined,
    })
    .where(eq(users.id, user!.id));

  return Response.json({ ok: true });
};

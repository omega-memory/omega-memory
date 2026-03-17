import type { APIRoute } from "astro";
import { getDb } from "../../../../lib/db";
import { generateWebAuthnAuthenticationOptions } from "../../../../lib/auth/providers/webauthn";

export const POST: APIRoute = async (context) => {
  const db = await getDb(context);

  try {
    const options = await generateWebAuthnAuthenticationOptions(db);

    return new Response(JSON.stringify(options), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

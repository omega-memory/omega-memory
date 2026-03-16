import type { APIRoute } from "astro";
import { getDb } from "../../lib/db";
import { AuthEngine, setProSessionCookie } from "../../lib/auth";

export const POST: APIRoute = async (context) => {
  const contentType = context.request.headers.get("content-type") ?? "";
  let licenseKey: string;

  if (contentType.includes("application/json")) {
    const body = await context.request.json();
    licenseKey = body.key ?? body.license_key ?? "";
  } else {
    const form = await context.request.formData();
    licenseKey = (form.get("key") as string) ?? (form.get("license_key") as string) ?? "";
  }

  if (!licenseKey) {
    return new Response(JSON.stringify({ error: "License key is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const db = await getDb(context);
  const secret =
    (context.locals.runtime as any)?.env?.AUTH_SECRET ??
    process.env.AUTH_SECRET ??
    "dev-secret";
  const auth = new AuthEngine(db, secret);

  const license = await auth.validateLicense(licenseKey);
  if (!license) {
    return new Response(JSON.stringify({ error: "Invalid or inactive license key" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const sessionToken = await auth.createProSession(licenseKey);
  if (!sessionToken) {
    return new Response(JSON.stringify({ error: "Failed to create session" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const cookie = setProSessionCookie(sessionToken);

  return new Response(
    JSON.stringify({
      ok: true,
      license: {
        id: license.id,
        email: license.email,
        status: license.status,
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Set-Cookie": cookie,
      },
    }
  );
};

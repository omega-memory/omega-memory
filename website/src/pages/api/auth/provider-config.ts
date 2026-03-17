import type { APIRoute } from "astro";
import { getAuthProvider } from "../../../lib/auth/factory";

export const GET: APIRoute = async () => {
  const provider = getAuthProvider();
  return new Response(JSON.stringify({
    provider: provider.name,
    methods: provider.getLoginMethods(),
  }), {
    headers: { "Content-Type": "application/json" },
  });
};

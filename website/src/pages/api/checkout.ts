import type { APIRoute } from "astro";

export const POST: APIRoute = async (context) => {
  const STRIPE_SECRET_KEY = import.meta.env.STRIPE_SECRET_KEY;

  if (!STRIPE_SECRET_KEY) {
    return new Response(JSON.stringify({ error: "Stripe is not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  let plan = "monthly";
  try {
    const formData = await context.request.formData();
    plan = (formData.get("plan") as string) || "monthly";
  } catch {
    // Fall back to JSON body
    try {
      const body = await context.request.json();
      plan = body.plan || "monthly";
    } catch {
      // Default to monthly
    }
  }

  const priceId =
    plan === "yearly"
      ? import.meta.env.STRIPE_PRICE_YEARLY || "price_yearly_placeholder"
      : import.meta.env.STRIPE_PRICE_MONTHLY || "price_monthly_placeholder";

  try {
    const response = await fetch("https://api.stripe.com/v1/checkout/sessions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${STRIPE_SECRET_KEY}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        "mode": "subscription",
        "success_url": `${new URL(context.request.url).origin}/pro/dashboard?session_id={CHECKOUT_SESSION_ID}`,
        "cancel_url": `${new URL(context.request.url).origin}/pricing`,
        "line_items[0][price]": priceId,
        "line_items[0][quantity]": "1",
      }).toString(),
    });

    const session = await response.json();

    if (!response.ok) {
      return new Response(JSON.stringify({ error: session.error?.message || "Failed to create checkout session" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    return context.redirect(session.url, 303);
  } catch (err) {
    return new Response(JSON.stringify({ error: "Failed to create checkout session" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

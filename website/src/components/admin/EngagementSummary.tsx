import React, { Suspense, lazy } from "react";
import type { XAccount } from "./lib/types";

const SocialAnalytics = lazy(() => import("./SocialAnalytics"));

function SocialAnalyticsLoading() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-8 w-40 rounded-lg bg-surface-elevated" />
      <div className="grid grid-cols-4 gap-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-20 rounded-xl bg-surface-elevated" />
        ))}
      </div>
    </div>
  );
}

export default function EngagementSummary({ account = "all" }: { account?: XAccount }) {
  return (
    <Suspense fallback={<SocialAnalyticsLoading />}>
      <SocialAnalytics account={account} />
    </Suspense>
  );
}

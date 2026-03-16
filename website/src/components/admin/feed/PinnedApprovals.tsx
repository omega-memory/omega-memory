import { useCallback, useEffect, useState } from "react";
import ApprovalCard, { type TweetItem, type ProposalItem } from "./ApprovalCard";

type UnifiedItem =
  | { type: "tweet"; item: TweetItem; date: string }
  | { type: "proposal"; item: ProposalItem; date: string };

interface PinnedApprovalsProps {
  refreshKey?: number;
}

export default function PinnedApprovals({ refreshKey }: PinnedApprovalsProps) {
  const [items, setItems] = useState<UnifiedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    const unified: UnifiedItem[] = [];
    let tweetFailed = false;
    let proposalFailed = false;

    // Fetch pending tweets and pending proposals in parallel
    const [tweetRes, proposalRes] = await Promise.allSettled([
      fetch("/api/tweets?status=pending"),
      fetch("/api/admin/orchestrator-feed?risk_level=high&status=active&limit=50"),
    ]);

    // Process tweets
    if (tweetRes.status === "fulfilled" && tweetRes.value.ok) {
      try {
        const data = await tweetRes.value.json();
        for (const t of data.tweets || []) {
          unified.push({
            type: "tweet",
            item: {
              id: t.id,
              type: t.type || "tweet",
              text: t.text,
              status: t.status,
              platform: t.platform || "x",
              x_account: t.x_account,
              created_at: t.created_at,
              scheduled_for: t.scheduled_for,
              source_tweet_text: t.source_tweet_text,
              source_author_handle: t.source_author_handle,
            },
            date: t.created_at || t.scheduled_for || "",
          });
        }
      } catch {
        tweetFailed = true;
      }
    } else {
      tweetFailed = true;
    }

    // Process proposals
    if (proposalRes.status === "fulfilled" && proposalRes.value.ok) {
      try {
        const data = await proposalRes.value.json();
        for (const p of data.items || []) {
          unified.push({
            type: "proposal",
            item: {
              id: p.id,
              item_type: p.item_type,
              risk_level: p.risk_level,
              status: p.status,
              title: p.title,
              body: p.body,
              metadata: p.metadata || {},
              source: p.source,
              created_at: p.created_at,
            },
            date: p.created_at || "",
          });
        }
      } catch {
        proposalFailed = true;
      }
    } else {
      proposalFailed = true;
    }

    // If both failed, show error
    if (tweetFailed && proposalFailed) {
      setError("Failed to load pending approvals");
      setItems([]);
      setLoading(false);
      return;
    }

    setError(null);

    // Sort newest first
    unified.sort((a, b) => {
      const da = a.date ? new Date(a.date).getTime() : 0;
      const db = b.date ? new Date(b.date).getTime() : 0;
      return db - da;
    });

    setItems(unified);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Re-fetch when parent signals new content
  useEffect(() => {
    if (refreshKey && refreshKey > 0) {
      fetchAll();
    }
  }, [refreshKey, fetchAll]);

  function handleAction(id: string) {
    setItems((prev) => prev.filter((entry) => entry.item.id !== id));
  }

  // Loading skeleton
  if (loading) {
    return (
      <div className="px-5 pt-4 pb-4 space-y-2">
        {[0, 1].map((i) => (
          <div key={i} className="admin-card p-4 space-y-2 animate-pulse" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="flex items-center gap-2">
              <div className="w-20 h-5 rounded bg-surface-elevated" />
              <div className="w-16 h-4 rounded bg-surface-elevated ml-auto" />
            </div>
            <div className="w-3/4 h-5 rounded bg-surface-elevated" />
            <div className="w-full h-10 rounded bg-surface-elevated" />
          </div>
        ))}
      </div>
    );
  }

  // Error state: both fetches failed
  if (error) {
    return (
      <div className="px-5 pt-4 pb-4">
        <div className="admin-card p-4 flex items-center justify-between border-l-[3px] border-l-type-error">
          <span className="text-[14px] text-type-error">{error}</span>
          <button
            onClick={() => { setLoading(true); setError(null); fetchAll(); }}
            className="px-3 py-1.5 rounded-lg text-[14px] font-medium text-ink-secondary hover:text-ink border border-edge hover:border-edge-strong transition-colors min-h-[36px] touch-manipulation"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Nothing pending: render nothing
  if (items.length === 0) {
    return null;
  }

  const count = items.length;

  return (
    <div>
      {/* Section header */}
      <div className="px-5 pt-4 pb-2 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-amber animate-pulse" />
        <span className="text-[14px] font-mono text-amber tracking-wider uppercase">
          {count} {count === 1 ? "item needs" : "items need"} attention
        </span>
      </div>

      {/* Card list */}
      <div className="px-5 pb-4 space-y-2">
        {items.map((entry) => (
          <ApprovalCard
            key={entry.item.id}
            type={entry.type}
            item={entry.item}
            onAction={handleAction}
          />
        ))}
      </div>

      {/* Divider */}
      <div className="mx-5 border-b border-edge-subtle" />
    </div>
  );
}

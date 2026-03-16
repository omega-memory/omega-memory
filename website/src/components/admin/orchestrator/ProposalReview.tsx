import { useCallback, useEffect, useState } from "react";
import OrchestratorCard from "./OrchestratorCard";
import OmegaMark from "./OmegaMark";
import type { OrchestratorFeedItem } from "../lib/orchestrator-feed";

interface ProposalReviewProps {
  onCountChange: (count: number) => void;
}

export default function ProposalReview({ onCountChange }: ProposalReviewProps) {
  const [items, setItems] = useState<OrchestratorFeedItem[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchProposals = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/orchestrator-feed?risk_level=high&status=active&limit=50");
      if (!res.ok) throw new Error(`${res.status}`);
      const json = await res.json();
      const proposals: OrchestratorFeedItem[] = json.items || [];
      setItems(proposals);
      onCountChange(proposals.length);
    } catch {
      onCountChange(0);
    }
    setLoading(false);
  }, [onCountChange]);

  useEffect(() => {
    fetchProposals();
  }, [fetchProposals]);

  async function handleApprove(id: string, reviewNote: string) {
    const item = items.find((i) => i.id === id);
    if (!item) return;

    try {
      const eventParams = (item.metadata as Record<string, unknown>).event_params || {};
      const res = await fetch(`/api/admin/orchestrator-feed/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve", event_params: eventParams, review_note: reviewNote || undefined }),
      });
      if (res.ok) {
        setItems((prev) => prev.filter((i) => i.id !== id));
        onCountChange(Math.max(0, items.length - 1));
      }
    } catch {
      // silent
    }
  }

  async function handleReject(id: string, reviewNote: string) {
    try {
      const res = await fetch(`/api/admin/orchestrator-feed/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reject", review_note: reviewNote || undefined }),
      });
      if (res.ok) {
        setItems((prev) => prev.filter((i) => i.id !== id));
        onCountChange(Math.max(0, items.length - 1));
      }
    } catch {
      // silent
    }
  }

  if (loading) {
    return (
      <div className="px-5 pt-2 space-y-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="admin-card p-4 space-y-2" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="flex items-center gap-2">
              <div className="skeleton w-16 h-5 rounded" />
              <div className="skeleton w-20 h-5 rounded" />
              <div className="skeleton w-16 h-4 rounded ml-auto" />
            </div>
            <div className="skeleton w-3/4 h-5 rounded" />
            <div className="skeleton w-full h-12 rounded" />
            <div className="flex gap-2">
              <div className="skeleton w-20 h-9 rounded-lg" />
              <div className="skeleton w-20 h-9 rounded-lg" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="text-center px-8 py-16">
        <div className="relative w-16 h-16 mx-auto mb-5">
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className="w-8 h-8 text-gold/30" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
            </svg>
          </div>
        </div>
        <p className="text-[16px] text-ink-tertiary">No proposals waiting for approval</p>
        <p className="text-[14px] text-ink-faint mt-1">High-risk actions will appear here for review</p>
      </div>
    );
  }

  return (
    <div className="px-5 pt-2 space-y-2 admin-stagger">
      {items.map((item, i) => (
        <OrchestratorCard
          key={item.id}
          item={item}
          onApprove={handleApprove}
          onReject={handleReject}
          style={{ animationDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

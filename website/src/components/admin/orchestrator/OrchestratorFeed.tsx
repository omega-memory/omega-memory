import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import OrchestratorCard from "./OrchestratorCard";
import OmegaMark from "./OmegaMark";
import type { OrchestratorFeedItem, OrchestratorItemType } from "../lib/orchestrator-feed";

// ── Visual config (mirrors OrchestratorCard for compact rows) ────

const COMPACT_CONFIG: Record<string, { color: string; label: string }> = {
  update:  { color: "var(--color-type-task)",     label: "Update" },
  insight: { color: "var(--color-type-lesson)",   label: "Insight" },
};

function formatTimeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Filter config ────────────────────────────────────────────────

const FILTERS: { label: string; value: string }[] = [
  { label: "All", value: "all" },
  { label: "Updates", value: "update" },
  { label: "Insights", value: "insight" },
  { label: "Reports", value: "report" },
  { label: "Alerts", value: "alert" },
  { label: "Proposals", value: "proposal" },
];

// ── Date grouping ────────────────────────────────────────────────

function getDateGroup(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (24 * 3600000));

  if (diffDays === 0) {
    // Check if actually today
    if (date.toDateString() === now.toDateString()) return "Today";
    return "Yesterday";
  }
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "This Week";
  if (diffDays < 30) return "This Month";
  return date.toLocaleDateString("en-US", { month: "long", year: "numeric" });
}

function groupByDate(items: OrchestratorFeedItem[]): [string, OrchestratorFeedItem[]][] {
  const groups = new Map<string, OrchestratorFeedItem[]>();
  for (const item of items) {
    const group = getDateGroup(item.created_at);
    const list = groups.get(group) || [];
    list.push(item);
    groups.set(group, list);
  }
  return Array.from(groups.entries());
}

// ── Component ────────────────────────────────────────────────────

export default function OrchestratorFeed() {
  const [items, setItems] = useState<OrchestratorFeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [hasMore, setHasMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const fetchMoreRef = useRef<() => void>(() => {});

  const fetchItems = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      params.set("limit", "30");
      if (filter !== "all") params.set("item_type", filter);

      const res = await fetch(`/api/admin/orchestrator-feed?${params}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const json = await res.json();

      setItems(json.items || []);
      setHasMore(json.hasMore ?? false);
      setNextCursor(json.nextCursor ?? null);
    } catch {
      // silent
    }
    setLoading(false);
    setRefreshing(false);
  }, [filter]);

  const fetchMore = useCallback(async () => {
    if (loadingMore || !hasMore || !nextCursor) return;
    setLoadingMore(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", "30");
      params.set("cursor", nextCursor);
      if (filter !== "all") params.set("item_type", filter);

      const res = await fetch(`/api/admin/orchestrator-feed?${params}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const json = await res.json();

      const newItems: OrchestratorFeedItem[] = json.items || [];
      if (newItems.length > 0) {
        setItems((prev) => [...prev, ...newItems]);
      }
      setHasMore(json.hasMore ?? false);
      setNextCursor(json.nextCursor ?? null);
    } catch {
      // silent
    }
    setLoadingMore(false);
  }, [loadingMore, hasMore, nextCursor, filter]);

  useEffect(() => {
    fetchMoreRef.current = fetchMore;
  }, [fetchMore]);

  // Infinite scroll observer
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) fetchMoreRef.current(); },
      { rootMargin: "200px" }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchItems();
  }, [fetchItems]);

  // ── Actions ────────────────────────────────────────────────────

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
        const { item: updated } = await res.json();
        setItems((prev) => prev.map((i) => (i.id === id ? updated : i)));
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
        const { item: updated } = await res.json();
        setItems((prev) => prev.map((i) => (i.id === id ? updated : i)));
      }
    } catch {
      // silent
    }
  }

  async function handleMarkRead(id: string) {
    try {
      const res = await fetch(`/api/admin/orchestrator-feed/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "read" }),
      });
      if (res.ok) {
        const { item: updated } = await res.json();
        setItems((prev) => prev.map((i) => (i.id === id ? updated : i)));
      }
    } catch {
      // silent
    }
  }

  async function handleMarkAllRead() {
    const unread = items.filter((i) => i.status !== "read" && i.status !== "approved" && i.status !== "rejected");
    if (unread.length === 0) return;
    // Optimistic: mark all as read locally
    setItems((prev) =>
      prev.map((i) =>
        unread.some((u) => u.id === i.id) ? { ...i, status: "read" as const } : i
      )
    );
    // Fire mutations in parallel
    unread.forEach((i) => {
      fetch(`/api/admin/orchestrator-feed/${i.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "read" }),
      }).catch(() => {});
    });
  }

  // ── Date-grouped items ─────────────────────────────────────────

  const grouped = useMemo(() => groupByDate(items), [items]);

  // ── Render ─────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="px-5 pt-4 space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="admin-card p-4 space-y-2" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="flex items-center gap-2">
              <div className="skeleton w-16 h-5 rounded" />
              <div className="skeleton w-12 h-5 rounded" />
              <div className="skeleton w-16 h-4 rounded ml-auto" />
            </div>
            <div className="skeleton w-3/4 h-5 rounded" />
            <div className="skeleton w-1/2 h-4 rounded" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="pb-4">
      {/* Filters */}
      <div className="flex items-center gap-2 px-5 pt-3 pb-2 overflow-x-auto scrollbar-hide filter-scroll-fade">
        {FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={`px-3 py-1.5 text-[13px] font-medium rounded-full whitespace-nowrap transition-all touch-manipulation ${
              filter === f.value
                ? "bg-gold/15 text-gold"
                : "text-ink-tertiary hover:text-ink-secondary"
            }`}
          >
            {f.label}
          </button>
        ))}
        {items.some((i) => i.status !== "read" && i.status !== "approved" && i.status !== "rejected") && (
          <button
            onClick={handleMarkAllRead}
            className="ml-auto text-[12px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation font-medium shrink-0"
          >
            Mark all read
          </button>
        )}
        <button
          onClick={() => { setRefreshing(true); fetchItems(); }}
          className={`${items.every((i) => i.status === "read" || i.status === "approved" || i.status === "rejected") ? "ml-auto" : ""} shrink-0 p-1.5 text-ink-tertiary hover:text-ink-secondary transition-colors touch-manipulation`}
          aria-label="Refresh"
        >
          <svg
            className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`}
            fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
          </svg>
        </button>
      </div>

      {/* Empty state */}
      {items.length === 0 ? (
        <div className="text-center px-8 py-16">
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div className="absolute inset-0 flex items-center justify-center">
              <OmegaMark size={40} className="text-gold/20 animate-breathe" />
            </div>
            <div className="ring-pulse" />
            <div className="ring-pulse" style={{ animationDelay: "0.8s" }} />
          </div>
          <p className="text-[15px] text-ink-tertiary">
            {filter === "all"
              ? "No orchestrator activity yet. Updates will appear as events run."
              : "Nothing matches this filter."}
          </p>
        </div>
      ) : (
        <div className="px-5 space-y-1">
          {grouped.map(([group, groupItems]) => (
            <div key={group}>
              <div className="pt-3 pb-2 sticky top-0 z-[5] bg-canvas/80 backdrop-blur-md flex items-center gap-3">
                <h3 className="text-[14px] font-semibold text-ink-secondary uppercase tracking-[0.08em]">
                  {group}
                </h3>
                <div className="flex-1 h-px bg-gradient-to-r from-edge to-transparent" />
              </div>
              <div className="space-y-1">
                {groupItems.map((item, i) => {
                  // Compact row for low-risk, non-actionable items
                  const isCompact = item.risk_level === "low"
                    && ["update", "insight"].includes(item.item_type)
                    && item.status !== "active";
                  if (isCompact) {
                    const cfg = COMPACT_CONFIG[item.item_type] || COMPACT_CONFIG.update;
                    return (
                      <div
                        key={item.id}
                        className="flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-surface-hover transition-colors"
                        style={{ animationDelay: `${i * 30}ms` }}
                      >
                        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: cfg.color }} />
                        <span className="text-[11px] font-semibold uppercase tracking-wider shrink-0" style={{ color: cfg.color }}>
                          {cfg.label}
                        </span>
                        <span className="text-[15px] text-ink truncate flex-1 min-w-0">
                          {item.title}
                        </span>
                        {item.account && (
                          <span className="text-[13px] text-ink-faint shrink-0 hidden sm:inline">@{item.account}</span>
                        )}
                        <span className="text-[12px] text-ink-faint font-mono shrink-0">
                          {formatTimeAgo(item.created_at)}
                        </span>
                        <button
                          onClick={() => handleMarkRead(item.id)}
                          className="text-[12px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation shrink-0"
                        >
                          Dismiss
                        </button>
                      </div>
                    );
                  }
                  return (
                    <OrchestratorCard
                      key={item.id}
                      item={item}
                      onApprove={handleApprove}
                      onReject={handleReject}
                      onMarkRead={handleMarkRead}
                      style={{ animationDelay: `${i * 50}ms` }}
                    />
                  );
                })}
              </div>
            </div>
          ))}

          {/* Infinite scroll sentinel */}
          <div ref={sentinelRef} className="h-1" />

          {/* Loading more */}
          {loadingMore && (
            <div className="space-y-2 pt-2 pb-4">
              {[0, 1].map((i) => (
                <div key={i} className="admin-card p-4 space-y-2">
                  <div className="skeleton w-3/4 h-5 rounded" />
                  <div className="skeleton w-1/2 h-4 rounded" />
                </div>
              ))}
            </div>
          )}

          {/* End of feed */}
          {!hasMore && items.length > 0 && (
            <p className="text-center text-[13px] text-ink-faint py-6">
              You've reached the beginning
            </p>
          )}
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseContentRich, timeAgo } from "./contentUtils";
import OmegaMark from "./OmegaMark";

import type { Memory } from "./feed/feedScoring";
import { getExpansionMode } from "./feed/feedScoring";
import { useFeedData } from "./feed/useFeedData";
import { useFeedKeyboard } from "./feed/useFeedKeyboard";
import FeedToolbar from "./feed/FeedToolbar";
import FeedRow from "./feed/FeedRow";
import FeedDetail from "./feed/FeedDetail";
import FeedInlineDetail from "./feed/FeedInlineDetail";
import FeedSummary from "./feed/FeedSummary";
import PinnedApprovals from "./feed/PinnedApprovals";
import SessionRecapCard from "./feed/SessionRecapCard";
import MissedNextSteps from "./feed/MissedNextSteps";
import { useBulkSelection } from "./hooks/useBulkSelection";
import BulkActionBar from "./shared/BulkActionBar";

// ─── Component ──────────────────────────────────────────────

interface FeedProps {
  onNavigateToApprovals?: () => void;
}

export default function Feed({ onNavigateToApprovals }: FeedProps) {
  const feed = useFeedData();
  const { focusedIndex, allVisibleIds } = useFeedKeyboard({
    important: feed.important,
    grouped: feed.grouped,
    memories: feed.memories,
    expandedId: feed.expandedId,
    setExpandedId: feed.setExpandedId,
    toggleExpand: feed.toggleExpand,
    handleDismiss: feed.handleDismiss,
    handleArchive: feed.handleArchive,
    filter: feed.filter,
    sortBy: feed.sortBy,
  });

  // ─── View mode ─────────────────────────────────────────
  const [viewMode, setViewMode] = useState<"timeline" | "summary">("timeline");

  // ─── Bulk selection ───────────────────────────────────────
  const visibleIds = useMemo(() => {
    const ids: string[] = [];
    feed.important.forEach((m: any) => ids.push(m.id));
    feed.grouped.forEach(([, items]: [string, any[]]) => items.forEach((m: any) => ids.push(m.id)));
    return ids;
  }, [feed.important, feed.grouped]);

  const bulk = useBulkSelection(visibleIds);

  const handleBulkArchive = useCallback(async () => {
    const ids = [...bulk.selectedIds];
    try {
      await Promise.all(ids.map((id) =>
        fetch(`/api/admin/memories/${id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ archived: true }),
        })
      ));
      bulk.clear();
      feed.handleRefresh();
    } catch (err) {
      console.error("[Feed] bulk archive failed:", err);
    }
  }, [bulk, feed]);

  // ─── Keyboard shortcut help ─────────────────────────────
  const [showShortcuts, setShowShortcuts] = useState(false);

  useEffect(() => {
    function handleQuestionMark(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "?") {
        e.preventDefault();
        setShowShortcuts((p) => !p);
      }
    }
    window.addEventListener("keydown", handleQuestionMark);
    return () => window.removeEventListener("keydown", handleQuestionMark);
  }, []);

  // ─── Scroll & Pull-to-Refresh ─────────────────────────────

  const [showScrollTop, setShowScrollTop] = useState(false);
  const [pullDistance, setPullDistance] = useState(0);
  const [refreshFlash, setRefreshFlash] = useState(false);
  const feedRef = useRef<HTMLDivElement>(null);
  const pullStartY = useRef(0);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const fetchMoreRef = useRef<() => void>(() => {});

  useEffect(() => {
    const onScroll = () => setShowScrollTop(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

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
    fetchMoreRef.current = feed.fetchMore;
  }, [feed.fetchMore]);

  const scrollToTop = useCallback(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    if (window.scrollY <= 0) pullStartY.current = e.touches[0].clientY;
  }, []);
  const onTouchMove = useCallback((e: React.TouchEvent) => {
    if (pullStartY.current > 0) {
      const dist = Math.max(0, e.touches[0].clientY - pullStartY.current);
      if (dist > 0 && dist < 150) setPullDistance(dist);
    }
  }, []);
  const onTouchEnd = useCallback(() => {
    if (pullDistance > 60) {
      feed.handleRefresh();
      // Show brief "Updated" flash after refresh settles
      setTimeout(() => {
        setRefreshFlash(true);
        setTimeout(() => setRefreshFlash(false), 1500);
      }, 600);
    }
    setPullDistance(0);
    pullStartY.current = 0;
  }, [pullDistance, feed.handleRefresh]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Auto-load more when client filter yields no results ──
  useEffect(() => {
    if (
      feed.isClientFiltering &&
      feed.filteredMemories.length === 0 &&
      feed.hasMore &&
      !feed.loadingMore &&
      !feed.loading
    ) {
      feed.fetchMore();
    }
  }, [feed.isClientFiltering, feed.filteredMemories.length, feed.hasMore, feed.loadingMore, feed.loading, feed.fetchMore]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Row renderer ─────────────────────────────────────────

  // Extract session achievements for MissedNextSteps
  const missedAchievements = feed.sessionAchievements
    .filter((m) => m.metadata?.session_achievement)
    .map((m) => {
      const sa = m.metadata!.session_achievement as { session_id: string; project: string | null; next_steps: string[] | null; has_unread_next_steps: boolean };
      return { ...sa, next_steps: Array.isArray(sa.next_steps) ? sa.next_steps : [] };
    });

  const renderRow = (m: Memory, isNeedsAttention = false, hideTimestamp = false) => {
    const isExpanded = feed.expandedId === m.id;
    const isRead = !!m.metadata?.read_at;
    const parsed = parseContentRich(m.content);
    const expMode = getExpansionMode(m, parsed);
    const isFocused = allVisibleIds[focusedIndex] === m.id;
    const isSessionRecap = !!m.metadata?.is_session_recap;
    return (
      <div key={m.id} data-memory-id={m.id}>
        <FeedRow
          m={m}
          isExpanded={isExpanded}
          onToggleExpand={feed.toggleExpand}
          isNeedsAttention={isNeedsAttention}
          isRead={isRead}
          expansionMode={expMode}
          density={feed.density}
          isFocused={isFocused}
          hideTimestamp={hideTimestamp}
        />
        {isExpanded && isSessionRecap && m.metadata?.session_achievement && (
          <SessionRecapCard
            achievement={m.metadata.session_achievement}
            isOpen={true}
            onClose={() => feed.setExpandedId(null)}
            onMarkNextStepRead={feed.handleMarkNextStepRead}
          />
        )}
        {isExpanded && !isSessionRecap && expMode === "full" && (
          <FeedDetail
            m={m}
            isOpen={true}
            onClose={() => feed.setExpandedId(null)}
            onNavigateToApprovals={onNavigateToApprovals}
            reminderActing={feed.reminderActing}
            onDismiss={feed.handleDismiss}
            onExtend={feed.handleExtend}
            onInteract={feed.handleInteract}
            onArchive={feed.handleArchive}
            onUnarchive={feed.handleUnarchive}
            confidentialUnlocked={feed.confidentialUnlocked}
            onUnlockRequest={feed.handleConfidentialUnlock}
          />
        )}
        {isExpanded && !isSessionRecap && expMode === "inline" && (
          <FeedInlineDetail
            m={m}
            isOpen={true}
            onArchive={feed.handleArchive}
            onDismiss={feed.handleDismiss}
          />
        )}
      </div>
    );
  };

  // ─── Loading skeleton ─────────────────────────────────────

  if (feed.loading) {
    return (
      <div role="status" aria-live="polite" aria-label="Loading memories">
        <div className="flex gap-2 px-5 pt-4 pb-2">
          {[56, 72, 64, 48, 60].map((w, i) => (
            <div key={i} className="skeleton h-[32px] rounded-full" style={{ width: w }} />
          ))}
        </div>
        <div className="space-y-1 px-5 pt-4">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg" style={{ animationDelay: `${i * 80}ms` }}>
              <div className="skeleton w-1.5 h-1.5 rounded-full" />
              <div className="skeleton w-4 h-4 rounded" />
              <div className="skeleton h-5 flex-1 rounded" />
              <div className="skeleton w-16 h-4 rounded" />
              <div className="skeleton w-8 h-4 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ─── Render ───────────────────────────────────────────────

  return (
    <div
      ref={feedRef}
      className="pb-4 relative"
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
    >
      {/* Pull to refresh indicator */}
      {pullDistance > 0 && (
        <div
          className="flex items-center justify-center overflow-hidden transition-all"
          style={{ height: pullDistance * 0.5 }}
        >
          <svg
            className={`w-5 h-5 text-ink-tertiary transition-transform ${pullDistance > 60 ? "scale-110" : ""}`}
            style={{ transform: `rotate(${Math.min(pullDistance * 3, 360)}deg)` }}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
          </svg>
        </div>
      )}

      <>
      {/* Pinned approvals (tweets + proposals needing action) */}
      <PinnedApprovals refreshKey={feed.approvalRefreshKey} />

      {/* Toolbar */}
      <FeedToolbar
        filter={feed.filter}
        onFilterChange={feed.setFilter}
        sortBy={feed.sortBy}
        onSortChange={feed.setSortBy}
        density={feed.density}
        onDensityChange={feed.setDensity}
        refreshing={feed.refreshing}
        onRefresh={feed.handleRefresh}
        onIngestUrl={feed.handleIngestUrl}
        showArchived={feed.showArchived}
        onShowArchivedChange={feed.setShowArchived}
        selectedProject={feed.selectedProject}
        onProjectChange={feed.setSelectedProject}
        availableProjects={feed.availableProjects}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      {viewMode === "summary" ? (
        <FeedSummary />
      ) : (
      <>
      {/* Keyboard shortcut hint */}
      <div className="px-5 pb-1 flex justify-end">
        <button
          onClick={() => setShowShortcuts((p) => !p)}
          className="text-[12px] text-ink-faint hover:text-ink-tertiary transition-colors flex items-center gap-1"
          title="Keyboard shortcuts"
        >
          <kbd className="px-1 py-0.5 rounded border border-edge text-[11px] font-mono">?</kbd>
          <span>Shortcuts</span>
        </button>
      </div>
      {showShortcuts && (
        <div className="mx-5 mb-3 p-3 bg-surface-elevated border border-edge rounded-lg text-[13px] text-ink-secondary">
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">j</kbd> / <kbd className="px-1 rounded border border-edge text-[11px] font-mono">k</kbd> Navigate</span>
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">Enter</kbd> Expand</span>
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">a</kbd> Archive</span>
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">d</kbd> Dismiss reminder</span>
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">Esc</kbd> Close / unfocus</span>
            <span><kbd className="px-1 rounded border border-edge text-[11px] font-mono">?</kbd> Toggle this help</span>
          </div>
        </div>
      )}

      {/* Error */}
      {feed.error && (
        <div
          className="mx-5 mb-3 p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[15px] text-type-error"
          role="alert"
        >
          {feed.error}
        </div>
      )}

      {/* Empty state */}
      {feed.filteredMemories.length === 0 ? (
        <div className="text-center px-8 py-16">
          {feed.isClientFiltering && feed.hasMore ? (
            <>
              <div className="flex items-center justify-center gap-2 mb-3">
                <svg className="animate-spin w-4 h-4 text-ink-tertiary" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <p className="text-[16px] text-ink-tertiary">
                  No matching items in this page, loading more...
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="relative w-20 h-20 mx-auto mb-6">
                <div className="absolute inset-0 flex items-center justify-center">
                  <OmegaMark size={40} className="text-gold/20 animate-breathe" />
                </div>
                <div className="ring-pulse" />
                <div className="ring-pulse" style={{ animationDelay: "0.8s" }} />
                <div className="ring-pulse" style={{ animationDelay: "1.6s" }} />
              </div>
              <p className="text-[16px] text-ink-tertiary">
                {feed.filter === "all" && feed.selectedProject === "all"
                  ? "Nothing here yet. Activity will show up as you work."
                  : feed.selectedProject !== "all"
                  ? `No items for this project${feed.filter !== "all" ? " and filter" : ""}`
                  : "Nothing matches this filter"}
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="px-5">
          {/* Missed next steps from sessions */}
          <MissedNextSteps
            achievements={missedAchievements}
            onMarkRead={feed.handleMarkNextStepRead}
          />

          {/* Needs Attention section */}
          {feed.important.length > 0 && (
            <div className="mb-2">
              <div className="pt-3 pb-2 flex items-center gap-2 sticky top-0 z-[5] bg-canvas/80 backdrop-blur-md">
                <span className="w-2 h-2 rounded-full bg-gold shadow-[0_0_8px_rgba(212,168,67,0.5)]" />
                <h3 className="text-[14px] font-semibold text-gold uppercase tracking-[0.1em]">
                  Needs Attention
                </h3>
                <span className="text-[14px] text-gold/60 font-medium tabular-nums">{feed.important.length}</span>
                <div className="flex-1 h-px bg-gradient-to-r from-gold/10 to-transparent ml-1" />
                {feed.important.some((m) => !m.metadata?.read_at) && (
                  <button
                    onClick={feed.handleMarkAllRead}
                    className="text-[14px] text-ink-faint hover:text-ink-tertiary transition-colors touch-manipulation font-medium"
                  >
                    Mark all read
                  </button>
                )}
              </div>
              <div className={feed.density === "compact" ? "space-y-0" : "space-y-0.5"}>
                {feed.important.map((m, idx) => {
                  const prev = idx > 0 ? timeAgo(feed.important[idx - 1].created_at) : null;
                  return renderRow(m, true, prev === timeAgo(m.created_at));
                })}
              </div>
            </div>
          )}

          {/* Main feed with date groups */}
          {feed.important.length > 0 && feed.displayItems.length > 0 && (
            <div className="h-px bg-gradient-to-r from-edge via-edge/50 to-transparent my-2" />
          )}
          {feed.grouped.map(([day, items]) => (
            <div key={day}>
              <div className="pt-3 pb-2 sticky top-0 z-[5] bg-canvas/80 backdrop-blur-md flex items-center gap-3">
                <h3 className="text-[18px] font-semibold text-ink tracking-tight">
                  {day}
                </h3>
                <div className="flex-1 h-px bg-gradient-to-r from-edge to-transparent" />
              </div>
              <div className={feed.density === "compact" ? "space-y-0" : "space-y-0.5"}>
                {items.map((m, idx) => {
                  const prev = idx > 0 ? timeAgo(items[idx - 1].created_at) : null;
                  return renderRow(m, false, prev === timeAgo(m.created_at));
                })}
              </div>
            </div>
          ))}

          {/* Infinite scroll sentinel */}
          <div ref={sentinelRef} className="h-1" />

          {/* Loading more skeleton */}
          {feed.loadingMore && (
            <div className="space-y-1 pt-2 pb-4">
              {[0, 1].map((i) => (
                <div key={i} className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg">
                  <div className="skeleton w-1.5 h-1.5 rounded-full" />
                  <div className="skeleton w-4 h-4 rounded" />
                  <div className="skeleton h-5 flex-1 rounded" />
                  <div className="skeleton w-12 h-4 rounded" />
                </div>
              ))}
            </div>
          )}

          {/* End of feed */}
          {!feed.hasMore && feed.memories.length > 0 && (
            <p className="text-center text-[14px] text-ink-faint py-6">
              You've reached the beginning
            </p>
          )}
        </div>
      )}

      {/* Bulk action bar */}
      <BulkActionBar count={bulk.count} onClear={bulk.clear}>
        <button
          onClick={handleBulkArchive}
          className="px-3 py-1.5 text-[13px] font-medium rounded-lg bg-ink-faint/10 text-ink-secondary hover:bg-ink-faint/20 transition-colors"
        >
          Archive selected
        </button>
      </BulkActionBar>
      </>
      )}

      </>

      {/* Scroll to top button */}
      <button
        onClick={scrollToTop}
        aria-label="Scroll to top"
        className={`fixed bottom-20 right-5 w-11 h-11 rounded-full bg-surface-elevated border border-edge text-ink-secondary hover:text-ink hover:border-edge-strong flex items-center justify-center shadow-card transition-all duration-200 touch-manipulation z-20 ${
          showScrollTop
            ? "opacity-100 translate-y-0"
            : "opacity-0 translate-y-2 pointer-events-none"
        }`}
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" />
        </svg>
      </button>

      {/* Pull-to-refresh confirmation toast */}
      <div
        className={`fixed top-4 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-surface-elevated border border-edge shadow-card text-[13px] text-ink-secondary flex items-center gap-1.5 transition-all duration-300 z-30 ${
          refreshFlash
            ? "opacity-100 translate-y-0"
            : "opacity-0 -translate-y-2 pointer-events-none"
        }`}
      >
        <svg className="w-3.5 h-3.5 text-green-500" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
        </svg>
        Updated
      </div>
    </div>
  );
}

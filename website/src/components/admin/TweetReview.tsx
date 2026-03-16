import React, { useCallback, useEffect, useState, Suspense, lazy } from "react";
import SlideDrawer from "./SlideDrawer";

const TargetsManager = lazy(() => import("./TargetsManager"));

/* ─── Unified approval item ─── */
type Platform = "x" | "reddit" | "linkedin" | "email";

function PlatformBadge({ platform, size = 14 }: { platform: Platform; size?: number }) {
  switch (platform) {
    case "x":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className="text-ink-secondary shrink-0">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
        </svg>
      );
    case "reddit":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className="text-[#FF4500] shrink-0">
          <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm6.066 13.71c.147.307.216.637.191.972-.073.986-.695 1.881-1.655 2.502C15.56 17.81 13.832 18.14 12 18.14s-3.56-.33-4.602-.956c-.96-.621-1.582-1.516-1.655-2.502a2.11 2.11 0 0 1 .191-.972c-.58-.313-.97-.902-.97-1.577 0-.997.83-1.808 1.85-1.808.496 0 .944.194 1.28.508 1.033-.634 2.399-1.027 3.906-1.087l.75-3.507.01-.038a.39.39 0 0 1 .464-.303l2.47.527c.2-.468.66-.798 1.196-.798.724 0 1.312.579 1.312 1.293s-.588 1.293-1.312 1.293c-.694 0-1.259-.543-1.306-1.225l-2.17-.463-.623 2.916c1.448.085 2.755.479 3.747 1.1.336-.314.784-.508 1.28-.508 1.02 0 1.85.811 1.85 1.808 0 .675-.39 1.264-.97 1.577zM9.063 13.4c0 .657.537 1.19 1.2 1.19.662 0 1.2-.533 1.2-1.19 0-.658-.538-1.19-1.2-1.19-.663 0-1.2.532-1.2 1.19zm5.674 1.19c.663 0 1.2-.533 1.2-1.19 0-.658-.537-1.19-1.2-1.19-.662 0-1.2.532-1.2 1.19 0 .657.538 1.19 1.2 1.19zm-1.082 1.58c-.32.32-.948.606-1.655.606s-1.335-.287-1.655-.607a.364.364 0 0 1 .515-.515c.199.199.66.413 1.14.413.48 0 .941-.214 1.14-.413a.364.364 0 0 1 .515.515z" />
        </svg>
      );
    case "linkedin":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className="text-[#0A66C2] shrink-0">
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
        </svg>
      );
    case "email":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" strokeWidth={1.5} stroke="currentColor" className="text-ink-secondary shrink-0">
          <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 0 1-2.25 2.25h-15a2.25 2.25 0 0 1-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0 0 19.5 4.5h-15a2.25 2.25 0 0 0-2.25 2.25m19.5 0v.243a2.25 2.25 0 0 1-1.07 1.916l-7.5 4.615a2.25 2.25 0 0 1-2.36 0L3.32 8.91a2.25 2.25 0 0 1-1.07-1.916V6.75" />
        </svg>
      );
  }
}

interface ReviewItem {
  id: string;
  type: "tweet" | "reply" | "reply_alert";
  text: string;
  label: string;
  date: string;
  status: string;
  platform: Platform;
  // tweet-specific
  x_account?: string;
  x_post_url?: string | null;
  content_type?: string;
  length_category?: string;
  image_suggestion?: string | null;
  image_url?: string | null;
  created_at?: string;
  // reply-specific
  source_tweet_text?: string;
  source_author_handle?: string;
  source_tweet_url?: string;
  algorithmic_value?: string;
  reply_type?: string;
  // reply-alert-specific
  our_tweet_text?: string;
  priority?: string;
  // reply queue
  scheduled_send_at?: string | null;
}

interface Props {
  onCountChange: (count: number) => void;
  refreshKey?: number;
  account?: "all" | "jasonsosa" | "omega_memory";
}

export default function TweetReview({ onCountChange, refreshKey, account = "all" }: Props) {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actingId, setActingId] = useState<string | null>(null);
  const [generatingImageId, setGeneratingImageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [regeneratingId, setRegeneratingId] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [targetsOpen, setTargetsOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<ReviewItem[]>([]);
  const [historyFilter, setHistoryFilter] = useState<"all" | "tweets" | "replies">("all");

  const load = useCallback(async () => {
    setError(null);
    const results: ReviewItem[] = [];
    const acctParam = account !== "all" ? `&account=${account}` : "";

    try {
      // Fetch ALL pending items in parallel: tweets, reply suggestions, reply alerts
      const [tweetRes, sugRes, alertRes] = await Promise.all([
        fetch(`/api/tweets?status=pending${acctParam}`),
        fetch(`/api/engagement?type=suggestions&status=pending${acctParam}`),
        fetch(`/api/engagement?type=alerts&status=pending${acctParam}`),
      ]);

      // Tweets
      if (tweetRes.ok) {
        const data = await tweetRes.json();
        for (const t of data.tweets || []) {
          results.push({
            id: t.id,
            type: "tweet",
            text: t.text,
            label: "Tweet",
            date: t.scheduled_for,
            status: t.status,
            platform: "x",
            x_account: t.x_account,
            x_post_url: t.x_post_url,
            content_type: t.content_type,
            length_category: t.length_category,
            image_suggestion: t.image_suggestion,
            image_url: t.image_url,
            created_at: t.created_at,
          });
        }
      }

      // Reply suggestions
      if (sugRes.ok) {
        const data = await sugRes.json();
        for (const s of data.suggestions || []) {
          results.push({
            id: s.id,
            type: "reply",
            text: s.suggested_reply,
            label: s.reply_type === "value_add" ? "Value Add" : s.reply_type === "experience_share" ? "Experience" : s.reply_type === "constructive_challenge" ? "Challenge" : "Conversation",
            date: s.created_at,
            status: s.status,
            platform: "x",
            x_account: s.x_account,
            source_tweet_text: s.source_tweet_text,
            source_author_handle: s.source_author_handle,
            source_tweet_url: s.source_tweet_url,
            algorithmic_value: s.algorithmic_value,
            reply_type: s.reply_type,
            scheduled_send_at: s.scheduled_send_at,
          });
        }
      }

      // Reply alerts (highest priority — prepend)
      if (alertRes.ok) {
        const data = await alertRes.json();
        for (const a of data.alerts || []) {
          results.unshift({
            id: a.id,
            type: "reply_alert",
            text: a.suggested_response,
            label: "Reply-to-Reply",
            date: a.created_at,
            status: a.status,
            platform: "x",
            x_account: a.x_account,
            our_tweet_text: a.our_tweet_text,
            source_tweet_text: a.reply_tweet_text,
            source_author_handle: a.reply_author_handle,
            algorithmic_value: "75x multiplier - respond within 30 min",
            priority: a.priority,
          });
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    }

    setItems(results);
    setLoading(false);
  }, [account]);

  useEffect(() => {
    setLoading(true);
    load();
  }, [load]);

  // Re-fetch when parent signals new content (after generation)
  useEffect(() => {
    if (refreshKey && refreshKey > 0) {
      setLoading(true);
      load();
    }
  }, [refreshKey, load]);

  // Report count to parent safely via effect, not during render/setState
  useEffect(() => {
    onCountChange(items.length);
  }, [items, onCountChange]);

  const loadHistory = useCallback(async () => {
    const acctParam = account !== "all" ? `&account=${account}` : "";
    const results: ReviewItem[] = [];
    try {
      const [tweetRes, sentRes] = await Promise.all([
        fetch(`/api/tweets?${account !== "all" ? `account=${account}` : ""}`),  // all statuses
        fetch(`/api/engagement?type=suggestions&status=sent${acctParam}`),
      ]);
      if (tweetRes.ok) {
        const data = await tweetRes.json();
        for (const t of data.tweets || []) {
          if (t.status === "pending") continue; // skip pending (already in queue)
          results.push({
            id: t.id, type: "tweet", text: t.text, label: "Tweet",
            date: t.scheduled_for, status: t.status, platform: "x",
            x_account: t.x_account, x_post_url: t.x_post_url,
            content_type: t.content_type, image_url: t.image_url,
          });
        }
      }
      if (sentRes.ok) {
        const data = await sentRes.json();
        for (const s of data.suggestions || []) {
          results.push({
            id: `reply-${s.id}`, type: "reply", text: s.suggested_reply, label: "Reply",
            date: s.sent_at || s.created_at, status: "sent", platform: "x",
            x_account: s.x_account, source_author_handle: s.source_author_handle,
            source_tweet_url: s.source_tweet_url, x_post_url: s.x_reply_url,
          });
        }
      }
    } catch { /* non-critical */ }
    setHistoryItems(results);
  }, [account]);

  // Load history when drawer opens
  useEffect(() => {
    if (historyOpen) loadHistory();
  }, [historyOpen, loadHistory]);

  function removeItem(id: string) {
    setItems((prev) => prev.filter((i) => i.id !== id));
  }

  async function handleApprove(item: ReviewItem) {
    setActingId(item.id);
    setError(null);
    try {
      let deferred = false;
      if (item.type === "tweet") {
        const res = await fetch(`/api/tweets/${item.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "approve" }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Publish failed (${res.status})`);
        }
        const data = await res.json();
        if (data.deferred) deferred = true;
      } else if (item.type === "reply") {
        const res = await fetch("/api/engagement", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "send", id: item.id }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Send failed (${res.status})`);
        }
      } else if (item.type === "reply_alert") {
        const res = await fetch("/api/engagement", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "approve_alert", id: item.id }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Approve failed (${res.status})`);
        }
      }
      removeItem(item.id);
      const label = item.type === "tweet" ? (deferred ? "Scheduled for publishing" : "Published to X") : "Approved";
      setToast(label);
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Approve failed");
    } finally {
      setActingId(null);
    }
  }

  async function handleReject(item: ReviewItem) {
    setActingId(item.id);
    setError(null);
    try {
      if (item.type === "tweet") {
        const res = await fetch(`/api/tweets/${item.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "reject" }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Reject failed (${res.status})`);
        }
      } else if (item.type === "reply") {
        const res = await fetch("/api/engagement", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "reject", id: item.id }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Reject failed (${res.status})`);
        }
      } else if (item.type === "reply_alert") {
        const res = await fetch("/api/engagement", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "reject_alert", id: item.id }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Reject failed (${res.status})`);
        }
      }
      removeItem(item.id);
      setToast("Rejected");
      setTimeout(() => setToast(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reject failed");
    } finally {
      setActingId(null);
    }
  }

  async function handleRegenerate(item: ReviewItem) {
    setRegeneratingId(item.id);
    setError(null);
    try {
      await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reject" }),
      });
      removeItem(item.id);
      setToast("Rejected - generate new content from the header");
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Regenerate failed");
    } finally {
      setRegeneratingId(null);
    }
  }

  async function handleGenerateImage(item: ReviewItem) {
    setGeneratingImageId(item.id);
    setError(null);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "image", tweetId: item.id }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Image generation failed (${res.status})`);
      }
      const { imageUrl } = await res.json();
      setItems((prev) =>
        prev.map((i) => (i.id === item.id ? { ...i, image_url: imageUrl } : i))
      );
      setToast("Image generated");
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image generation failed");
    } finally {
      setGeneratingImageId(null);
    }
  }

  async function handleRemoveImage(item: ReviewItem) {
    try {
      await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "edit", text: item.text, image_url: null }),
      });
      setItems((prev) =>
        prev.map((i) => (i.id === item.id ? { ...i, image_url: null } : i))
      );
    } catch { /* non-critical */ }
  }

  async function handleToggleAccount(item: ReviewItem) {
    const newAccount = item.x_account === "omega_memory" ? "jasonsosa" : "omega_memory";
    try {
      const res = await fetch(`/api/tweets/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "set_account", x_account: newAccount }),
      });
      if (!res.ok) throw new Error("Failed to switch account");
      setItems((prev) =>
        prev.map((i) => (i.id === item.id ? { ...i, x_account: newAccount } : i))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to switch account");
    }
  }

  async function handleRecycle(item: ReviewItem) {
    setActingId(item.id);
    setError(null);
    try {
      const res = await fetch("/api/tweets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: item.text, x_account: item.x_account }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Failed to recycle tweet");
      }
      setToast("Recycled to queue");
      setTimeout(() => setToast(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recycle failed");
    } finally {
      setActingId(null);
    }
  }

  function formatDate(dateStr: string) {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }

  function formatTimeAgo(dateStr: string) {
    const d = new Date(dateStr);
    const mins = Math.floor((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return formatDate(dateStr);
  }

  if (loading) {
    return (
      <div role="status" aria-live="polite" aria-label="Loading review items">
        <div className="flex items-center justify-between px-5 pt-4 pb-3">
          <div className="skeleton h-4 w-12 rounded-md" />
          <div className="flex gap-2">
            <div className="skeleton h-8 w-20 rounded-lg" />
            <div className="skeleton h-8 w-20 rounded-lg" />
          </div>
        </div>
        <div className="space-y-3 px-5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="admin-card overflow-hidden relative" style={{ animationDelay: `${i * 120}ms` }}>
              <div className="h-[3px] skeleton" />
              <div className="p-4 space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="skeleton h-5 w-14 rounded-md" />
                  <div className="skeleton h-4 w-20 rounded-md" />
                </div>
                <div className="skeleton h-5 w-4/5 rounded-md" />
                <div className="skeleton h-4 w-2/3 rounded-md" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="pb-4">
      {/* Header: Social label + secondary view buttons */}
      <div className="flex items-center justify-between px-6 pt-5 pb-3">
        <h3 className="text-[16px] font-medium text-ink-secondary">Social</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setHistoryOpen(true)}
            className="flex items-center gap-1.5 px-3 py-2 text-[14px] font-medium text-ink-faint hover:text-ink-secondary rounded-lg border border-edge hover:bg-surface-hover transition-colors cursor-pointer"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
            </svg>
            History
          </button>
          <button
            onClick={() => setTargetsOpen(true)}
            className="flex items-center gap-1.5 px-3 py-2 text-[14px] font-medium text-ink-faint hover:text-ink-secondary rounded-lg border border-edge hover:bg-surface-hover transition-colors cursor-pointer"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1 1 15 0Z" />
            </svg>
            Targets
          </button>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="mx-5 mb-4 p-3.5 bg-type-lesson/[0.12] border border-type-lesson/25 rounded-xl text-[15px] font-medium text-type-lesson card-enter flex items-center gap-2.5" role="status">
          <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
          {toast}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mx-5 mb-4 p-3.5 bg-type-error/[0.12] border border-type-error/25 rounded-xl text-[15px] text-type-error flex items-center gap-2.5" role="alert">
          <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
          </svg>
          {error}
        </div>
      )}

      {/* Flat queue -- all actionable items */}
      {items.length === 0 ? (
        <div className="px-6 py-12 text-center text-[16px] text-ink-tertiary">
          All caught up
        </div>
      ) : (
        <div className="space-y-3 px-6 admin-stagger">
          {items.map((item, idx) => {
            const busy = actingId === item.id;
            const isGenImage = generatingImageId === item.id;
            const isTweet = item.type === "tweet";
            const isReply = item.type === "reply";
            const isAlert = item.type === "reply_alert";

            // Bar color based on type
            const barColor = isAlert ? "bg-type-error" : "bg-gold";

            return (
              <div
                key={item.id}
                className={`admin-card overflow-hidden card-enter relative ${
                  isAlert ? "!border-type-error/30" : ""
                }`}
                style={{ animationDelay: `${idx * 40}ms` }}
              >
                <div className={`absolute top-0 left-0 right-0 h-[3px] ${barColor}`} />
                <div className="p-4 pt-5">
                  {/* Card header -- varies by type */}
                  {isTweet ? (
                    <div className="flex items-center gap-2.5 mb-3">
                      <img src={item.x_account === "omega_memory" ? "https://pbs.twimg.com/profile_images/2025777526250094592/dI93Qeiu_normal.jpg" : (process.env.NEXT_PUBLIC_TWITTER_AVATAR_URL || "https://pbs.twimg.com/profile_images/1907680931160403968/pbBpKq_c_normal.jpg")} alt="" className="w-9 h-9 rounded-full shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[14px] font-semibold text-ink truncate">{item.x_account === "omega_memory" ? "OMEGA Memory" : (process.env.NEXT_PUBLIC_TWITTER_DISPLAY_NAME || "Jason Sosa")}</span>
                          <svg className="w-3.5 h-3.5 text-gold shrink-0" viewBox="0 0 24 24" fill="currentColor"><path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.998-3.818-3.998-.47 0-.92.084-1.336.25C14.818 2.415 13.51 1.5 12 1.5s-2.816.917-3.437 2.25c-.415-.165-.866-.25-1.336-.25-2.11 0-3.818 1.79-3.818 4 0 .494.083.964.237 1.4-1.272.65-2.147 2.018-2.147 3.6 0 1.495.782 2.798 1.942 3.486-.02.17-.032.34-.032.514 0 2.21 1.708 4 3.818 4 .47 0 .92-.086 1.335-.25.62 1.334 1.926 2.25 3.437 2.25 1.512 0 2.818-.916 3.437-2.25.415.163.865.248 1.336.248 2.11 0 3.818-1.79 3.818-4 0-.174-.012-.344-.033-.513 1.158-.687 1.943-1.99 1.943-3.484zm-6.616-3.334l-4.334 6.5c-.145.217-.382.334-.625.334-.143 0-.288-.04-.416-.126l-.115-.094-2.415-2.415c-.293-.293-.293-.768 0-1.06s.768-.294 1.06 0l1.77 1.767 3.825-5.74c.23-.345.696-.436 1.04-.207.346.23.44.696.21 1.04z" /></svg>
                        </div>
                        <span className="text-[14px] text-ink-faint">@{item.x_account === "omega_memory" ? "omega_memory" : (process.env.NEXT_PUBLIC_TWITTER_HANDLE || "jasonsosa")}</span>
                      </div>
                      <button
                        onClick={() => handleToggleAccount(item)}
                        className="shrink-0 px-3 py-2 rounded-lg border border-edge hover:border-gold/30 hover:bg-gold/[0.06] text-ink-faint hover:text-gold text-[14px] font-medium transition-all flex items-center gap-1.5 touch-manipulation cursor-pointer"
                        title={`Switch to @${item.x_account === "omega_memory" ? "jasonsosa" : "omega_memory"}`}
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                        </svg>
                        Switch
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2.5 mb-3">
                      <PlatformBadge platform={item.platform} />
                      <span className={`text-[14px] font-semibold px-2.5 py-1 rounded-md uppercase tracking-wider ${
                        isAlert ? "bg-type-error/10 text-type-error" : "bg-gold/10 text-gold"
                      }`}>
                        {item.label}
                      </span>
                      {item.source_author_handle && (
                        <span className="text-[14px] text-ink-faint">@{item.source_author_handle}</span>
                      )}
                      <span className="text-[14px] text-ink-faint ml-auto font-medium">
                        via @{item.x_account === "omega_memory" ? "omega_memory" : "jasonsosa"}
                      </span>
                      <span className="text-[14px] text-ink-faint">{formatTimeAgo(item.date)}</span>
                    </div>
                  )}

                  {/* Source tweet (for replies/alerts) */}
                  {(isReply || isAlert) && item.source_tweet_text && (
                    <div className="mb-3 p-3.5 rounded-lg bg-ink-faint/5 border border-edge">
                      <p className="text-[15px] text-ink-secondary leading-relaxed line-clamp-2">
                        {isAlert && item.our_tweet_text ? (
                          <>
                            <span className="text-ink-faint">You: </span>
                            {item.our_tweet_text}
                            <br />
                            <span className="text-ink-faint">Reply: </span>
                            {item.source_tweet_text}
                          </>
                        ) : (
                          item.source_tweet_text
                        )}
                      </p>
                    </div>
                  )}

                  {/* Content text */}
                  <p className="text-[15px] text-ink leading-relaxed whitespace-pre-wrap mb-3">
                    {item.text}
                  </p>

                  {/* Image preview (tweets only) */}
                  {isTweet && item.image_url && (
                    <div className="mb-3 relative group/img">
                      <img src={item.image_url} alt="Tweet image" className="w-full rounded-xl border border-edge object-cover" style={{ aspectRatio: "16/9" }} />
                      <div className="absolute top-2 right-2 flex gap-1.5 opacity-0 group-hover/img:opacity-100 transition-opacity">
                        <button onClick={() => handleGenerateImage(item)} disabled={isGenImage} className="px-3 py-2 rounded-lg bg-canvas/80 backdrop-blur-sm border border-edge text-[14px] text-ink-secondary hover:text-ink transition-colors touch-manipulation cursor-pointer">
                          {isGenImage ? "..." : "Regenerate"}
                        </button>
                        <button onClick={() => handleRemoveImage(item)} className="px-3 py-2 rounded-lg bg-canvas/80 backdrop-blur-sm border border-edge text-[14px] text-ink-faint hover:text-type-error transition-colors touch-manipulation cursor-pointer">
                          Remove
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Image generation button */}
                  {isTweet && !item.image_url && item.image_suggestion && (
                    <button onClick={() => handleGenerateImage(item)} disabled={isGenImage} className="mb-3 w-full py-3 rounded-xl border border-dashed border-edge-subtle hover:border-gold/30 text-ink-faint hover:text-gold text-[14px] transition-all flex items-center justify-center gap-2 touch-manipulation disabled:opacity-50 cursor-pointer">
                      {isGenImage ? (
                        <>
                          <span className="w-3.5 h-3.5 border-2 border-gold/30 border-t-gold rounded-full animate-spin" />
                          <span className="text-gold">Generating image...</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z" />
                          </svg>
                          Generate image
                        </>
                      )}
                    </button>
                  )}

                  {/* Metadata (tweets) */}
                  {isTweet && (
                    <div className="flex items-center gap-2 mb-3 text-[14px] text-ink-faint font-mono">
                      <span>{formatDate(item.date)}</span>
                      {item.content_type && (<><span>&middot;</span><span>{item.content_type}</span></>)}
                      {item.length_category && (<><span>&middot;</span><span>{item.length_category}</span></>)}
                      {item.created_at && (<><span>&middot;</span><span>generated {new Date(item.created_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" })} ET</span></>)}
                    </div>
                  )}

                  {/* Algorithm value (replies) */}
                  {(isReply || isAlert) && item.algorithmic_value && (
                    <div className="text-[14px] text-ink-faint mb-3">{item.algorithmic_value}</div>
                  )}

                  {/* Scheduled send time */}
                  {item.scheduled_send_at && (
                    <div className="text-[14px] font-medium text-gold mb-3 flex items-center gap-1.5">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                      </svg>
                      Sending at {new Date(item.scheduled_send_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" })} ET
                    </div>
                  )}
                  {isAlert && item.status === "approved" && (
                    <div className="text-[14px] font-semibold text-type-error mb-3 flex items-center gap-1.5">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                      </svg>
                      Sending ASAP
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2.5 items-center">
                    <button
                      onClick={() => handleApprove(item)}
                      disabled={busy}
                      className={`px-5 py-2.5 rounded-lg text-canvas text-[14px] font-semibold transition-all disabled:opacity-50 touch-manipulation ${
                        isAlert ? "bg-type-error hover:bg-type-error/80" : "bg-type-lesson hover:bg-type-lesson/80"
                      }`}
                    >
                      {busy ? "..." : isAlert ? "Respond" : isTweet ? "Approve" : isReply ? "Send" : "Approve"}
                    </button>
                    <button
                      onClick={() => handleReject(item)}
                      disabled={busy}
                      className="px-5 py-2.5 rounded-lg border border-edge hover:bg-surface-hover text-ink-secondary text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation"
                    >
                      {isAlert ? "Dismiss" : "Reject"}
                    </button>
                    {isTweet && (
                      <button
                        onClick={() => handleRegenerate(item)}
                        disabled={busy || regeneratingId === item.id}
                        className="px-5 py-2.5 rounded-lg border border-gold/20 hover:bg-gold/[0.06] text-gold text-[14px] font-medium transition-colors disabled:opacity-50 touch-manipulation ml-auto"
                      >
                        {regeneratingId === item.id ? "..." : "Regen"}
                      </button>
                    )}
                    {(isReply || isAlert) && item.source_tweet_url && (
                      <a href={item.source_tweet_url} target="_blank" rel="noopener noreferrer" className="ml-auto px-3 py-2 text-[14px] text-ink-faint hover:text-ink-secondary transition-colors" onClick={(e) => e.stopPropagation()}>
                        View original
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* History drawer */}
      <SlideDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} title="Tweet History">
        <div className="px-5 pt-4">
          {/* Filter chips */}
          <div className="flex gap-1.5 mb-4">
            {(["all", "tweets", "replies"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setHistoryFilter(f)}
                className={`px-3.5 py-2 text-[14px] font-medium rounded-lg border transition-all touch-manipulation cursor-pointer ${
                  historyFilter === f
                    ? "bg-gold/[0.12] border-gold/20 text-gold"
                    : "border-edge text-ink-faint hover:text-ink-secondary"
                }`}
              >
                {f === "all" ? "All" : f === "tweets" ? "Tweets" : "Replies"}
              </button>
            ))}
          </div>

          {/* History items */}
          {(() => {
            const filtered = historyFilter === "all" ? historyItems
              : historyFilter === "tweets" ? historyItems.filter((i) => i.type === "tweet")
              : historyItems.filter((i) => i.type === "reply" || i.type === "reply_alert");

            if (filtered.length === 0) {
              return <div className="py-8 text-center text-[15px] text-ink-tertiary">No history yet</div>;
            }

            return (
              <div className="space-y-3">
                {filtered.map((item) => {
                  const isApproved = item.status === "published" || item.status === "approved";
                  const isRejected = item.status === "rejected";
                  const histBarColor = isApproved ? "bg-type-lesson" : isRejected ? "bg-type-error" : "bg-ink-faint";
                  return (
                    <div key={item.id} className="admin-card overflow-hidden relative">
                      <div className={`absolute top-0 left-0 right-0 h-[3px] ${histBarColor}`} />
                      <div className="p-4 pt-5 min-w-0">
                        <p className="text-[15px] text-ink leading-relaxed line-clamp-2 mb-2.5">{item.text}</p>
                        {item.image_url && (
                          <img src={item.image_url} alt="" className="w-full rounded-lg border border-edge object-cover mb-2.5" style={{ aspectRatio: "16/9", maxHeight: 160 }} />
                        )}
                        <div className="flex items-center gap-2.5 text-[14px]">
                          <PlatformBadge platform={item.platform} />
                          <span className="text-[14px] font-semibold px-2.5 py-1 rounded-md bg-ink-faint/10 text-ink-tertiary uppercase tracking-wider">{item.label}</span>
                          <span className={`font-medium ${isApproved ? "text-type-lesson" : isRejected ? "text-type-error" : "text-ink-tertiary"}`}>
                            {isApproved ? "Approved" : isRejected ? "Rejected" : item.status}
                          </span>
                          <span className="text-ink-faint">&middot;</span>
                          <span className="text-[14px] text-ink-faint font-medium">via @{item.x_account === "omega_memory" ? "omega_memory" : "jasonsosa"}</span>
                          <span className="text-ink-faint">&middot;</span>
                          <span className="text-ink-faint">{formatDate(item.date)}</span>
                          {item.type === "tweet" && isApproved && (
                            <div className="flex items-center gap-2 ml-auto">
                              <button onClick={() => handleRecycle(item)} disabled={actingId === item.id} className="text-ink-faint hover:text-gold transition-colors font-medium touch-manipulation disabled:opacity-50">
                                {actingId === item.id ? "..." : "Recycle"}
                              </button>
                              {item.x_post_url && (
                                <a href={item.x_post_url} target="_blank" rel="noopener noreferrer" className="text-gold hover:text-gold-dim transition-colors font-medium" onClick={(e) => e.stopPropagation()}>
                                  View &rarr;
                                </a>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>
      </SlideDrawer>

      {/* Targets drawer */}
      <SlideDrawer open={targetsOpen} onClose={() => setTargetsOpen(false)} title="Engagement Targets">
        <Suspense fallback={<div className="px-5 pt-6 animate-pulse"><div className="h-24 rounded-xl bg-surface-elevated" /></div>}>
          <TargetsManager />
        </Suspense>
      </SlideDrawer>
    </div>
  );
}

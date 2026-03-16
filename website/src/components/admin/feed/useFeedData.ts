import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  classifyContent,
  parseContentRich,
} from "../contentUtils";
import type { Memory, SortOption } from "./feedScoring";
import {
  splitByImportance,
  groupByDaySorted,
  sortByScore,
  PRIMARY_FILTERS,
  SECONDARY_FILTERS,
  isActionRequired,
  isInsight,
  isProgress,
  dismissReminder,
  extendReminder,
  markAsRead,
  archiveMemory,
  unarchiveMemory,
} from "./feedScoring";

const CONFIDENTIAL_SESSION_KEY = "omega-confidential-unlocked";

function buildSessionContent(a: Record<string, unknown>): string {
  const tasks = a.completed_tasks as string[] | undefined;
  const task = a.task as string | undefined;
  const title = task || (tasks && tasks.length > 0 ? tasks[0] : "Session completed");
  const taskCount = tasks ? tasks.length : 0;
  const filesCount = (a.files_modified as string[] | undefined)?.length || 0;
  const parts = [title];
  if (taskCount > 1) parts.push(`${taskCount} tasks completed`);
  if (filesCount > 0) parts.push(`${filesCount} files modified`);
  return parts.join(" - ");
}

/** Items that are pure system housekeeping with no value to a human reader */
function isFeedNoise(m: Memory): boolean {
  const content = m.content;
  const firstLine = content.split("\n")[0].trim().toLowerCase();

  // Coordination system noise: guard claims, file claims, hook invocations
  if (/^(pre-edit guard claim|auto-claimed on edit|file claim|hook invocation)\b/.test(firstLine)) return true;

  // Raw XML / internal task notification data
  if (/<task-id>|<tool-use-id>|<task-notification>/i.test(content)) return true;

  // "Fact:" followed by XML tags or internal IDs
  if (/^fact:\s*</i.test(firstLine)) return true;

  // Just bare file counts: "1 Files Modified", "3 files changed"
  if (/^\d+\s+files?\s+(modified|changed|updated)$/i.test(firstLine)) return true;

  // Internal tool output: raw JSON objects as content
  if (firstLine.startsWith("{") && content.includes('"tool_use_id"')) return true;

  // Session summaries that are just "Session completed" or bare file counts with no real info
  if (m.event_type === "session_summary") {
    const stripped = content.replace(/\s*[-–—]\s*\d+\s+(files?\s+(modified|changed)|tasks?\s+completed)/gi, "").trim();
    if (/^session completed$/i.test(stripped) || stripped.length < 30) return true;
  }

  // Auto-captured sessions with no real content ("agent made 0 omega_store calls")
  if (/auto-captured session/i.test(firstLine) && /\bagent made 0\b/i.test(content)) return true;

  // Any item whose entire content is just a short generic phrase
  if (content.trim().length < 20 && /^(session completed|coding|files? modified|update|fix|changes?)$/i.test(firstLine)) return true;

  return false;
}

export function useFeedData() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState<SortOption>("priority");
  const [refreshing, setRefreshing] = useState(false);
  const [approvalRefreshKey, setApprovalRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [reminderActing, setReminderActing] = useState<Set<string>>(new Set());
  const [density, setDensity] = useState<"default" | "compact">(() => {
    try {
      const stored = localStorage.getItem("omega-density");
      return stored === "compact" ? "compact" : "default";
    }
    catch { return "default"; }
  });
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [selectedProject, setSelectedProject] = useState("all");
  const initialPageSize = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const fetchMoreAbortRef = useRef<AbortController | null>(null);
  const refreshTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Confidential unlock
  const [confidentialUnlocked, setConfidentialUnlocked] = useState(() => {
    try {
      const stored = sessionStorage.getItem(CONFIDENTIAL_SESSION_KEY);
      return stored !== null;
    } catch { return false; }
  });

  const handleConfidentialUnlock = useCallback(async () => {
    try {
      const optRes = await fetch("/api/auth/webauthn/auth-options", { method: "POST" });
      if (!optRes.ok) { setError("Could not start authentication"); return; }
      const options = await optRes.json();
      const { startAuthentication } = await import("@simplewebauthn/browser");
      const authResp = await startAuthentication({ optionsJSON: options });
      const verRes = await fetch("/api/auth/webauthn/auth-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(authResp),
      });
      if (verRes.ok) {
        setConfidentialUnlocked(true);
        try { sessionStorage.setItem(CONFIDENTIAL_SESSION_KEY, String(Date.now())); } catch {}
      } else {
        setError("Authentication failed");
      }
    } catch (err) {
      if (err instanceof Error && err.name === "NotAllowedError") return;
      setError("Authentication failed");
    }
  }, []);

  useEffect(() => {
    try { localStorage.setItem("omega-density", density); } catch {}
  }, [density]);

  // ─── Data fetching ─────────────────────────────────────────

  // Outcome-based filters (action_required, insights, progress) are always client-side
  const OUTCOME_FILTERS = ["action_required", "insights", "progress"];
  const isClientFilter =
    SECONDARY_FILTERS.some((f) => f.value === filter) ||
    OUTCOME_FILTERS.includes(filter) ||
    selectedProject !== "all";
  const serverFilter = isClientFilter ? "all" :
    (PRIMARY_FILTERS.some((f) => f.value === filter) && !OUTCOME_FILTERS.includes(filter)) ? filter : "all";

  const fetchMemories = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setNextCursor(null);
    setHasMore(true);
    try {
      const params = new URLSearchParams();
      params.set("page_size", "50");
      if (serverFilter !== "all") params.set("event_type", serverFilter);

      const res = await fetch(`/api/memories?${params}`, { signal: controller.signal });
      if (!res.ok) throw new Error(`${res.status}`);

      const json = await res.json();
      const items: Memory[] = json.memories || [];
      setMemories(items);
      initialPageSize.current = items.length;
      setHasMore(json.hasMore ?? false);
      setNextCursor(json.nextCursor ?? null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setLoading(false);
        setRefreshing(false);
        return;
      }
      setError("Couldn't load memories. Check your connection.");
      setLoading(false);
      setRefreshing(false);
    }
  }, [serverFilter]);

  const fetchMore = useCallback(async () => {
    if (loadingMore || !hasMore || !nextCursor) return;
    fetchMoreAbortRef.current?.abort();
    const controller = new AbortController();
    fetchMoreAbortRef.current = controller;
    setLoadingMore(true);
    try {
      const params = new URLSearchParams();
      params.set("page_size", "50");
      params.set("cursor", nextCursor);
      if (serverFilter !== "all") params.set("event_type", serverFilter);

      const res = await fetch(`/api/memories?${params}`, { signal: controller.signal });
      if (!res.ok) throw new Error(`${res.status}`);

      const json = await res.json();
      const newMemories: Memory[] = json.memories || [];
      if (newMemories.length > 0) {
        setMemories((prev) => [...prev, ...newMemories]);
      }
      setHasMore(json.hasMore ?? false);
      setNextCursor(json.nextCursor ?? null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError("Failed to load more items");
    }
    setLoadingMore(false);
  }, [loadingMore, hasMore, nextCursor, serverFilter]);

  useEffect(() => {
    setLoading(true);
    fetchMemories();
    return () => {
      abortRef.current?.abort();
      fetchMoreAbortRef.current?.abort();
      if (refreshTimeoutRef.current) clearTimeout(refreshTimeoutRef.current);
    };
  }, [fetchMemories]);

  // Fetch session achievements after initial feed renders
  const [sessionAchievements, setSessionAchievements] = useState<Memory[]>([]);
  const hasFetchedAchievementsRef = useRef(false);
  useEffect(() => {
    if (loading || hasFetchedAchievementsRef.current) return;
    hasFetchedAchievementsRef.current = true;
    fetch("/api/admin/session-achievements?days=14")
      .then((r) => r.json())
      .then((data) => {
        if (data.achievements && data.achievements.length > 0) {
          const virtual: Memory[] = data.achievements.map((a: Record<string, unknown>) => ({
            id: `session-${a.session_id}`,
            content: buildSessionContent(a),
            event_type: "session_recap",
            priority: (a.has_unread_next_steps ? 4 : 2) as number,
            created_at: a.started_at as string,
            project: a.project as string | null,
            metadata: {
              is_session_recap: true,
              session_achievement: a,
            },
          }));
          setSessionAchievements(virtual);
        }
      })
      .catch(() => {});
  }, [loading]);

  // ─── Filtering ─────────────────────────────────────────────

  // Derive available projects from loaded memories
  const availableProjects = useMemo(() => {
    const slugs = new Set<string>();
    for (const m of memories) {
      if (m.project) {
        const slug = m.project.split("/").filter(Boolean).pop() || "";
        if (slug) slugs.add(slug);
      }
    }
    return Array.from(slugs).sort();
  }, [memories]);

  const filteredMemories = useMemo(() => {
    let result = [...memories, ...sessionAchievements];

    // ── Hide system noise that has no human value ──
    result = result.filter((m) => !isFeedNoise(m));

    if (!showArchived) {
      result = result.filter((m) => !m.metadata?.archived_at);
    }

    if (filter !== "reminder") {
      result = result.filter(
        (m) =>
          m.event_type !== "reminder" ||
          m.metadata?.reminder_status !== "dismissed"
      );
    }

    // Apply project filter first
    if (selectedProject !== "all") {
      result = result.filter((m) => {
        if (!m.project) return false;
        const slug = m.project.split("/").filter(Boolean).pop() || "";
        return slug === selectedProject;
      });
    }

    // Server-side filters (decision, error_pattern) already handled by API
    if (!isClientFilter && selectedProject === "all") return result;

    // Outcome-based filters
    if (filter === "action_required") {
      return result.filter(isActionRequired);
    }
    if (filter === "insights") {
      return result.filter(isInsight);
    }
    if (filter === "progress") {
      return result.filter(isProgress);
    }

    // Secondary filters
    if (filter === "deadlines") {
      return result.filter((m) => {
        const p = parseContentRich(m.content);
        return p.dueDate !== null;
      });
    }
    if (filter === "tagged") {
      return result.filter((m) => m.metadata?.tag);
    }
    if (filter === "grants") {
      return result.filter((m) => classifyContent(m.content) === "grant");
    }
    if (filter === "x_brief") {
      return result.filter((m) => {
        const c = m.content.toLowerCase();
        return c.startsWith("x brief") || c.includes("x brief ") || m.metadata?.tag === "x_brief";
      });
    }
    if (filter === "url_ingest") {
      return result.filter((m) => m.metadata?.tag === "url_ingest");
    }
    if (filter === "reminder") {
      return result.filter((m) => m.event_type === "reminder");
    }
    if (filter === "session_recap") {
      return result.filter((m) => m.event_type === "session_recap");
    }
    if (["lesson_learned", "task_completion", "user_preference"].includes(filter)) {
      return result.filter((m) => m.event_type === filter);
    }
    return result;
  }, [memories, sessionAchievements, filter, isClientFilter, showArchived, selectedProject]);

  // ─── Sorting & Grouping ────────────────────────────────────

  const { important, displayItems } = useMemo(() => {
    if (sortBy === "priority") {
      const firstPage = filteredMemories.slice(0, initialPageSize.current || filteredMemories.length);
      const { important, rest: firstPageRest } = splitByImportance(firstPage);
      const overflow = filteredMemories.slice(firstPage.length);
      const rest = sortByScore([...firstPageRest, ...overflow]);
      return { important, displayItems: rest };
    }
    if (sortBy === "unread") {
      const sorted = [...filteredMemories].sort((a, b) => {
        const aRead = a.metadata?.read_at ? 1 : 0;
        const bRead = b.metadata?.read_at ? 1 : 0;
        if (aRead !== bRead) return aRead - bRead; // unread first
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      return { important: [] as Memory[], displayItems: sorted };
    }
    const sorted = [...filteredMemories].sort((a, b) => {
      const da = new Date(a.created_at).getTime();
      const db = new Date(b.created_at).getTime();
      return sortBy === "newest" ? db - da : da - db;
    });
    return { important: [] as Memory[], displayItems: sorted };
  }, [filteredMemories, sortBy]);

  const grouped = useMemo(() => {
    return groupByDaySorted(displayItems);
  }, [displayItems]);

  // ─── Actions ───────────────────────────────────────────────

  function handleRefresh() {
    setRefreshing(true);
    setApprovalRefreshKey((k) => k + 1);
    fetchMemories();
  }

  function toggleExpand(id: string) {
    const expanding = expandedId !== id;
    setExpandedId((prev) => (prev === id ? null : id));

    if (expanding) {
      const m = memories.find((mem) => mem.id === id);
      if (m && !m.metadata?.read_at) {
        setMemories((prev) =>
          prev.map((mem) =>
            mem.id === id
              ? { ...mem, metadata: { ...(mem.metadata || {}), read_at: new Date().toISOString() } }
              : mem
          )
        );
        markAsRead(id, m.metadata);
      }
    }
  }

  async function handleDismiss(m: Memory) {
    setReminderActing((prev) => new Set(prev).add(m.id));
    const ok = await dismissReminder(m.id, m.metadata);
    if (ok) {
      setMemories((prev) =>
        prev.map((mem) =>
          mem.id === m.id
            ? { ...mem, metadata: { ...(mem.metadata || {}), reminder_status: "dismissed", dismissed_from: "web", dismissed_at: new Date().toISOString() } }
            : mem
        )
      );
    }
    setReminderActing((prev) => { const next = new Set(prev); next.delete(m.id); return next; });
  }

  async function handleExtend(m: Memory) {
    setReminderActing((prev) => new Set(prev).add(m.id));
    const { success, newRemindAt } = await extendReminder(m.id, m.metadata);
    if (success && newRemindAt) {
      setMemories((prev) =>
        prev.map((mem) =>
          mem.id === m.id
            ? { ...mem, metadata: { ...(mem.metadata || {}), remind_at: newRemindAt, extended_from: "web", extended_at: new Date().toISOString() } }
            : mem
        )
      );
    }
    setReminderActing((prev) => { const next = new Set(prev); next.delete(m.id); return next; });
  }

  async function handleArchive(m: Memory) {
    const originalMetadata = m.metadata;
    setMemories((prev) =>
      prev.map((mem) =>
        mem.id === m.id
          ? { ...mem, metadata: { ...(mem.metadata || {}), archived_at: new Date().toISOString(), archived_from: "web" } }
          : mem
      )
    );
    setExpandedId(null);
    try {
      await archiveMemory(m.id, m.metadata);
    } catch {
      setMemories((prev) =>
        prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: originalMetadata } : mem))
      );
    }
  }

  async function handleUnarchive(m: Memory) {
    const originalMetadata = m.metadata;
    const updated = { ...(m.metadata || {}) };
    delete updated.archived_at;
    delete updated.archived_from;
    setMemories((prev) =>
      prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: updated } : mem))
    );
    try {
      await unarchiveMemory(m.id, m.metadata);
    } catch {
      setMemories((prev) =>
        prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: originalMetadata } : mem))
      );
    }
  }

  async function handleInteract(m: Memory, payload: Record<string, any>) {
    setReminderActing((prev) => new Set(prev).add(m.id));
    const updated = { ...(m.metadata || {}), ...payload };
    setMemories((prev) =>
      prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: updated } : mem))
    );
    try {
      const res = await fetch("/api/memories", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: m.id, metadata: updated }),
      });
      if (!res.ok) {
        setMemories((prev) =>
          prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: m.metadata } : mem))
        );
      }
    } catch {
      setMemories((prev) =>
        prev.map((mem) => (mem.id === m.id ? { ...mem, metadata: m.metadata } : mem))
      );
    }
    setReminderActing((prev) => { const next = new Set(prev); next.delete(m.id); return next; });
  }

  function handleMarkAllRead() {
    const unread = important.filter((m) => !m.metadata?.read_at);
    if (unread.length === 0) return;
    const now = new Date().toISOString();
    setMemories((prev) =>
      prev.map((mem) => {
        if (unread.some((u) => u.id === mem.id)) {
          return { ...mem, metadata: { ...(mem.metadata || {}), read_at: now } };
        }
        return mem;
      })
    );
    // Skip synthetic session IDs that don't exist in the DB
    unread
      .filter((m) => !m.id.startsWith("session-"))
      .forEach((m) => markAsRead(m.id, m.metadata));
  }

  async function handleMarkNextStepRead(sessionId: string) {
    // Optimistically update local state
    setSessionAchievements((prev) =>
      prev.map((m) => {
        if (m.id !== `session-${sessionId}`) return m;
        const achievement = { ...m.metadata?.session_achievement, has_unread_next_steps: false };
        return {
          ...m,
          priority: 2,
          metadata: { ...m.metadata, session_achievement: achievement },
        };
      })
    );
    // Find the handoff and mark read
    try {
      // First get the handoff ID for this session
      const handoffRes = await fetch(`/api/admin/coordination/handoffs?session_id=${sessionId}`);
      if (handoffRes.ok) {
        const { handoffs } = await handoffRes.json();
        if (handoffs && handoffs.length > 0) {
          await fetch("/api/admin/coordination/handoffs", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "mark_read", handoff_id: handoffs[0].id }),
          });
        }
      }
    } catch {
      // Non-critical - optimistic update already applied
    }
  }

  async function handleIngestUrl(url: string): Promise<{ success: boolean; domain?: string; error?: string }> {
    try {
      const res = await fetch("/api/admin/ingest-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) return { success: false, error: data.error || "Ingestion failed" };
      fetchMemories();
      return { success: true, domain: data.domain };
    } catch {
      return { success: false, error: "Network error" };
    }
  }

  return {
    // State
    memories,
    loading,
    filter,
    setFilter,
    sortBy,
    setSortBy,
    refreshing,
    approvalRefreshKey,
    error,
    expandedId,
    setExpandedId,
    reminderActing,
    density,
    setDensity,
    loadingMore,
    hasMore,
    showArchived,
    setShowArchived,
    selectedProject,
    setSelectedProject,
    confidentialUnlocked,

    // Derived
    filteredMemories,
    availableProjects,
    important,
    displayItems,
    grouped,
    isClientFiltering: isClientFilter && filter !== "all",
    sessionAchievements,

    // Actions
    fetchMore,
    handleRefresh,
    toggleExpand,
    handleDismiss,
    handleExtend,
    handleArchive,
    handleUnarchive,
    handleInteract,
    handleMarkAllRead,
    handleIngestUrl,
    handleConfidentialUnlock,
    handleMarkNextStepRead,
  };
}

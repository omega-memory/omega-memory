import { FormEvent, useCallback, useEffect, useRef, useState } from "react";


interface Message {
  role: "user" | "assistant";
  content: string;
}

interface SearchResult {
  content: string;
  event_type?: string;
  similarity?: number | null;
  document_title?: string;
  source_path?: string;
}

type Mode = "search" | "chat" | "actions";

const CHAT_STORAGE_KEY = "omega-chat-history";
const RECENT_STORAGE_KEY = "omega-palette-recent";
const MAX_CHAT_HISTORY = 50;
const MAX_RECENT = 10;

const TYPE_COLORS: Record<string, string> = {
  decision: "bg-type-decision/15 text-type-decision",
  lesson_learned: "bg-type-lesson/15 text-type-lesson",
  user_preference: "bg-type-preference/15 text-type-preference",
  error_pattern: "bg-type-error/15 text-type-error",
  session_summary: "bg-type-session/15 text-type-session",
  task_completion: "bg-type-task/15 text-type-task",
  reminder: "bg-type-reminder/15 text-type-reminder",
};

const TYPE_LABELS: Record<string, string> = {
  decision: "Decision",
  lesson_learned: "Lesson",
  user_preference: "Preference",
  error_pattern: "Error",
  session_summary: "Session",
  task_completion: "Task",
  reminder: "Reminder",
};

function similarityBar(similarity: number): string {
  if (similarity >= 0.8) return "bg-type-lesson";
  if (similarity >= 0.6) return "bg-gold";
  if (similarity >= 0.4) return "bg-type-reminder";
  return "bg-ink-faint";
}

// Quick actions available in actions mode
interface QuickAction {
  id: string;
  label: string;
  description: string;
  icon: string;
  handler: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [mode, setMode] = useState<Mode>("search");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const stored: Message[] = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || "[]");
      return stored.length > MAX_CHAT_HISTORY ? stored.slice(stored.length - MAX_CHAT_HISTORY) : stored;
    } catch { return []; }
  });
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recentQueries, setRecentQueries] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      return JSON.parse(localStorage.getItem(RECENT_STORAGE_KEY) || "[]");
    } catch { return []; }
  });
  const [actionFocusIndex, setActionFocusIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const streamingContentRef = useRef("");
  const chatAbortRef = useRef<AbortController | null>(null);

  // Quick actions
  const quickActions: QuickAction[] = [
    {
      id: "nav-dashboard",
      label: "Go to Dashboard",
      description: "Navigate to the dashboard tab",
      icon: "M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6Z",
      handler: () => { navigateToTab("dashboard"); },
    },
    {
      id: "nav-feed",
      label: "Go to Feed",
      description: "Navigate to the activity feed",
      icon: "M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
      handler: () => { navigateToTab("feed"); },
    },
    {
      id: "nav-jobs",
      label: "Go to Jobs",
      description: "Navigate to scheduled jobs",
      icon: "M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5",
      handler: () => { navigateToTab("jobs"); },
    },
    {
      id: "nav-entities",
      label: "Go to Entities",
      description: "Navigate to entity management",
      icon: "M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21",
      handler: () => { navigateToTab("entities"); },
    },
    {
      id: "nav-coordination",
      label: "Go to Coordination",
      description: "Navigate to agent coordination",
      icon: "M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5",
      handler: () => { navigateToTab("coordination"); },
    },
    {
      id: "nav-settings",
      label: "Go to Settings",
      description: "Navigate to settings",
      icon: "M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281",
      handler: () => { navigateToTab("settings"); },
    },
  ];

  function navigateToTab(tab: string) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", tab);
    window.location.href = url.toString();
  }

  const filteredActions = input.trim()
    ? quickActions.filter((a) => a.label.toLowerCase().includes(input.toLowerCase()) || a.description.toLowerCase().includes(input.toLowerCase()))
    : quickActions;

  const saveRecentQuery = useCallback((query: string) => {
    setRecentQueries((prev) => {
      const filtered = prev.filter((q) => q !== query);
      const next = [query, ...filtered].slice(0, MAX_RECENT);
      localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setActionFocusIndex(-1);
    } else {
      setInput("");
      setError(null);
      chatAbortRef.current?.abort();
    }
  }, [open]);

  useEffect(() => {
    const toStore = messages.length > MAX_CHAT_HISTORY
      ? messages.slice(messages.length - MAX_CHAT_HISTORY)
      : messages;
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(toStore));
  }, [messages]);

  // Keyboard navigation for actions mode
  useEffect(() => {
    if (!open || mode !== "actions") return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActionFocusIndex((prev) => Math.min(prev + 1, filteredActions.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActionFocusIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter" && actionFocusIndex >= 0 && actionFocusIndex < filteredActions.length) {
        e.preventDefault();
        filteredActions[actionFocusIndex].handler();
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, mode, actionFocusIndex, filteredActions, onClose]);

  const detectPrefixSearch = useCallback(async (query: string) => {
    // Type-prefixed queries
    if (query.startsWith("@")) {
      const term = query.slice(1).trim();
      if (!term) return null;
      const res = await fetch(`/api/admin/entities?search=${encodeURIComponent(term)}&limit=10`);
      if (!res.ok) return null;
      const data = await res.json();
      return (data.entities || []).map((e: any) => ({
        content: `${e.name} (${e.entityType})${e.status ? ` - ${e.status}` : ""}`,
        event_type: "entity",
        similarity: null,
      }));
    }
    if (query.startsWith("#")) {
      const term = query.slice(1).trim();
      if (!term) return null;
      const res = await fetch("/api/admin/projects/overview");
      if (!res.ok) return null;
      const data = await res.json();
      const projects = (data.projects || []).filter((p: any) =>
        p.displayName?.toLowerCase().includes(term.toLowerCase()) ||
        p.project?.toLowerCase().includes(term.toLowerCase())
      );
      return projects.map((p: any) => ({
        content: `${p.displayName || p.project} - ${p.momentum || "unknown"} momentum, ${p.sessionCount || 0} sessions`,
        event_type: "project",
        similarity: null,
      }));
    }
    if (query.startsWith(">")) {
      const term = query.slice(1).trim();
      if (!term) return null;
      const res = await fetch("/api/schedules");
      if (!res.ok) return null;
      const data = await res.json();
      const schedules = (data.schedules || []).filter((s: any) =>
        s.name?.toLowerCase().includes(term.toLowerCase()) ||
        s.label?.toLowerCase().includes(term.toLowerCase())
      );
      return schedules.map((s: any) => ({
        content: `${s.name} [${s.label}] - ${s.enabled ? "enabled" : "disabled"}, last: ${s.last_status || "never"}`,
        event_type: "job",
        similarity: null,
      }));
    }
    return null;
  }, []);

  const handleSearch = useCallback(async (query: string) => {
    setLoading(true);
    setSearchResults([]);
    setError(null);
    saveRecentQuery(query);

    try {
      // Check for prefix search
      const prefixResults = await detectPrefixSearch(query);
      if (prefixResults) {
        setSearchResults(prefixResults);
        if (prefixResults.length === 0) {
          setError("No results found.");
        }
        setLoading(false);
        return;
      }

      const res = await fetch("/api/admin/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 10 }),
      });

      if (!res.ok) throw new Error(`Search failed (${res.status})`);

      const data = await res.json();
      const results = [
        ...(data.results?.memories || []).map((m: any) => ({
          content: m.content,
          event_type: m.event_type,
          similarity: m.similarity,
        })),
        ...(data.results?.documents || []).map((d: any) => ({
          content: d.content,
          document_title: d.document_title,
          source_path: d.source_path,
          similarity: d.similarity,
        })),
      ];
      setSearchResults(results);
      if (results.length === 0) {
        setError("No results found. Try different search terms.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed.");
    }
    setLoading(false);
  }, [saveRecentQuery, detectPrefixSearch]);

  const handleChat = useCallback(async (userMessage: string) => {
    if (loading) return;
    const newMessages: Message[] = [...messages, { role: "user", content: userMessage }];
    setMessages(newMessages);
    setLoading(true);
    setError(null);

    chatAbortRef.current?.abort();
    const controller = new AbortController();
    chatAbortRef.current = controller;

    try {
      const res = await fetch("/api/admin/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, history: newMessages.slice(-10) }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      streamingContentRef.current = "";
      setMessages([...newMessages, { role: "assistant", content: "" }]);

      if (reader) {
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (jsonStr === "[DONE]") continue;
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === "content_block_delta" && event.delta?.text) {
                streamingContentRef.current += event.delta.text;
                const content = streamingContentRef.current;
                setMessages([...newMessages, { role: "assistant", content }]);
              }
            } catch { /* skip */ }
          }
        }
      }

      const finalContent = streamingContentRef.current;
      setMessages([
        ...newMessages,
        { role: "assistant", content: finalContent || "No response received." },
      ]);
    } catch (err) {
      setMessages([
        ...newMessages,
        { role: "assistant", content: `Error: ${err instanceof Error ? err.message : "Unknown"}` },
      ]);
    }
    setLoading(false);
  }, [messages, loading]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const query = input.trim();
    if (!query) return;
    setInput("");
    if (mode === "search") handleSearch(query);
    else if (mode === "chat") handleChat(query);
    else if (mode === "actions") {
      // In actions mode, Enter on the input filters
    }
  }

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="command-palette-backdrop fixed inset-0 z-50 bg-canvas/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="command-palette fixed inset-0 z-50 flex items-start justify-center pt-[15vh] px-4 pointer-events-none">
        <div
          className="pointer-events-auto w-full max-w-[640px] max-h-[70vh] flex flex-col rounded-xl bg-surface border border-edge shadow-[0_25px_60px_rgba(0,0,0,0.5)] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          {/* Input area */}
          <form onSubmit={handleSubmit} className="flex items-center gap-2 px-4 py-3 border-b border-edge-subtle">
            <svg className="w-4 h-4 text-ink-faint shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
            </svg>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setActionFocusIndex(-1);
              }}
              placeholder={mode === "search" ? "Search memories... (@entity, #project, >job)" : mode === "chat" ? "Ask anything..." : "Quick action..."}
              className="flex-1 bg-transparent text-[14px] text-ink placeholder-ink-faint focus:outline-none"
              onKeyDown={(e) => {
                if (e.key === "Escape") onClose();
              }}
            />
            {/* Mode toggle */}
            <div className="flex gap-0.5 p-0.5 bg-surface-elevated rounded-md shrink-0">
              <button
                type="button"
                onClick={() => setMode("search")}
                className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                  mode === "search" ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-tertiary"
                }`}
              >
                Search
              </button>
              <button
                type="button"
                onClick={() => setMode("chat")}
                className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                  mode === "chat" ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-tertiary"
                }`}
              >
                Ask
              </button>
              <button
                type="button"
                onClick={() => setMode("actions")}
                className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                  mode === "actions" ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-tertiary"
                }`}
              >
                Actions
              </button>
            </div>
          </form>

          {/* Results */}
          <div ref={resultsRef} className="flex-1 overflow-y-auto px-4 py-2 space-y-2 scrollbar-hide">
            {error && (
              <div className="p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
                {error}
              </div>
            )}

            {mode === "search" ? (
              <>
                {/* Recent queries when input is empty */}
                {!loading && searchResults.length === 0 && !error && !input.trim() && recentQueries.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[11px] uppercase tracking-wider text-ink-faint/40 font-mono px-1 py-1">Recent</div>
                    {recentQueries.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => { setInput(q); handleSearch(q); }}
                        className="w-full text-left px-3 py-2 rounded-lg text-[13px] text-ink-secondary hover:bg-surface-hover transition-colors flex items-center gap-2"
                      >
                        <svg className="w-3.5 h-3.5 text-ink-faint shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                        </svg>
                        {q}
                      </button>
                    ))}
                  </div>
                )}

                {loading && (
                  <div className="space-y-2">
                    {[0, 1, 2].map((i) => (
                      <div key={i} className="rounded-lg bg-surface-elevated border border-edge p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="skeleton h-4 w-14 rounded-md" />
                          <div className="ml-auto skeleton h-1 w-12 rounded-full" />
                        </div>
                        <div className="skeleton h-3 w-full rounded mb-1" />
                        <div className="skeleton h-3 w-4/5 rounded" />
                      </div>
                    ))}
                  </div>
                )}
                {searchResults.map((r, i) => (
                  <div
                    key={i}
                    className="rounded-lg bg-surface-elevated border border-edge p-3 card-enter"
                    style={{ animationDelay: `${i * 30}ms` }}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      {r.event_type && (
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-medium ${
                          TYPE_COLORS[r.event_type] || "bg-ink-faint/20 text-ink-tertiary"
                        }`}>
                          {TYPE_LABELS[r.event_type] || r.event_type.replace(/_/g, " ")}
                        </span>
                      )}
                      {r.document_title && (
                        <span className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-type-decision/15 text-type-decision">
                          {r.document_title}
                        </span>
                      )}
                      {r.similarity != null && (
                        <div className="ml-auto flex items-center gap-1.5">
                          <div className="w-10 h-1 bg-surface rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${similarityBar(r.similarity)}`}
                              style={{ width: `${Math.round(r.similarity * 100)}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-ink-faint tabular-nums">
                            {(r.similarity * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                    <p className="text-[13px] text-ink-secondary leading-relaxed line-clamp-3">
                      {r.content}
                    </p>
                  </div>
                ))}
                {!loading && searchResults.length === 0 && !error && input.trim() === "" && recentQueries.length === 0 && (
                  <div className="text-center py-8 text-ink-tertiary text-[13px]">
                    Type to search your memories
                  </div>
                )}
              </>
            ) : mode === "actions" ? (
              /* Actions mode */
              <div className="space-y-1">
                {filteredActions.map((action, i) => (
                  <button
                    key={action.id}
                    onClick={() => { action.handler(); onClose(); }}
                    className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center gap-3 ${
                      i === actionFocusIndex
                        ? "bg-gold/[0.08] text-gold"
                        : "text-ink-secondary hover:bg-surface-hover"
                    }`}
                  >
                    <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d={action.icon} />
                    </svg>
                    <div className="min-w-0">
                      <div className="text-[14px] font-medium">{action.label}</div>
                      <div className="text-[12px] text-ink-faint">{action.description}</div>
                    </div>
                  </button>
                ))}
                {filteredActions.length === 0 && (
                  <div className="text-center py-8 text-ink-tertiary text-[13px]">
                    No matching actions
                  </div>
                )}
              </div>
            ) : (
              /* Chat mode */
              <>
                {messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-[13px] leading-relaxed ${
                      msg.role === "user"
                        ? "bg-surface-elevated text-ink border border-edge"
                        : "bg-surface text-ink-secondary border border-edge-subtle"
                    }`}>
                      {msg.content || (
                        <span className="inline-flex gap-1.5 py-0.5">
                          <span className="w-1.5 h-1.5 bg-ink-tertiary rounded-full animate-bounce" />
                          <span className="w-1.5 h-1.5 bg-ink-tertiary rounded-full animate-bounce [animation-delay:0.15s]" />
                          <span className="w-1.5 h-1.5 bg-ink-tertiary rounded-full animate-bounce [animation-delay:0.3s]" />
                        </span>
                      )}
                    </div>
                  </div>
                ))}
                {messages.length === 0 && !loading && (
                  <div className="text-center py-8 text-ink-tertiary text-[13px]">
                    Ask anything about your memories
                  </div>
                )}
              </>
            )}
          </div>

          {/* Footer hint */}
          <div className="flex items-center justify-between px-4 py-2 border-t border-edge-subtle text-[11px] text-ink-faint">
            <span>
              {mode === "chat" && messages.length > 0 && (
                <button
                  onClick={() => { setMessages([]); localStorage.removeItem(CHAT_STORAGE_KEY); }}
                  className="hover:text-ink-tertiary transition-colors"
                >
                  Clear history
                </button>
              )}
              {mode === "search" && recentQueries.length > 0 && (
                <button
                  onClick={() => { setRecentQueries([]); localStorage.removeItem(RECENT_STORAGE_KEY); }}
                  className="hover:text-ink-tertiary transition-colors"
                >
                  Clear recent
                </button>
              )}
            </span>
            <span>
              {mode === "search" && <span className="mr-2 text-ink-faint/40">@entity #project &gt;job</span>}
              <kbd className="font-mono">Esc</kbd> to close
            </span>
          </div>
        </div>
      </div>
    </>
  );
}

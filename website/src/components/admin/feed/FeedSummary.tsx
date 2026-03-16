import { useCallback, useEffect, useRef, useState } from "react";

interface FeedSummaryProps {
  /** Trigger a refresh */
  refreshKey?: number;
}

export default function FeedSummary({ refreshKey }: FeedSummaryProps) {
  const [summary, setSummary] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSummary(null);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/admin/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: "Summarize the most important recent activity. Group by project. Highlight anything that needs attention. Be concise, use bullet points.",
          history: [],
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let content = "";

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
                content += event.delta.text;
                setSummary(content);
              }
            } catch { /* skip */ }
          }
        }
      }

      setSummary(content || "No summary available.");
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError(err instanceof Error ? err.message : "Failed to generate summary");
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchSummary();
    return () => abortRef.current?.abort();
  }, [fetchSummary, refreshKey]);

  return (
    <div className="px-5 py-4">
      {loading && !summary && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-[13px] text-ink-tertiary">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
            </svg>
            Generating activity summary...
          </div>
          <div className="skeleton h-4 w-3/4 rounded" />
          <div className="skeleton h-4 w-full rounded" />
          <div className="skeleton h-4 w-2/3 rounded" />
        </div>
      )}
      {error && (
        <div className="p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
          {error}
          <button onClick={fetchSummary} className="ml-2 underline hover:no-underline">Retry</button>
        </div>
      )}
      {summary && (
        <div className="prose-sm">
          <div className="text-[14px] text-ink-secondary leading-relaxed whitespace-pre-wrap">
            {summary}
          </div>
          {loading && (
            <span className="inline-flex gap-1 ml-1">
              <span className="w-1 h-1 bg-ink-tertiary rounded-full animate-bounce" />
              <span className="w-1 h-1 bg-ink-tertiary rounded-full animate-bounce [animation-delay:0.15s]" />
              <span className="w-1 h-1 bg-ink-tertiary rounded-full animate-bounce [animation-delay:0.3s]" />
            </span>
          )}
        </div>
      )}
    </div>
  );
}

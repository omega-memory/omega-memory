import { useState, useRef, useCallback, useEffect } from "react";
import type { XAccount } from "../lib/types";

export function useGenerateAction(account: XAccount) {
  const [generating, setGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const safetyRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const feedbackRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const settleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollAbortRef = useRef<AbortController | null>(null);

  function cleanupGenPolling() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (safetyRef.current) { clearTimeout(safetyRef.current); safetyRef.current = null; }
    if (feedbackRef.current) { clearTimeout(feedbackRef.current); feedbackRef.current = null; }
    if (settleTimerRef.current) { clearTimeout(settleTimerRef.current); settleTimerRef.current = null; }
    pollAbortRef.current?.abort();
  }

  const handleGenerate = useCallback(async () => {
    if (generating) return; // prevent double-fire
    cleanupGenPolling();
    setGenerating(true);
    setGenProgress("Queued...");
    console.log("[generate] enqueuing job...");

    try {
      const res = await fetch("/api/generate-hourly", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: account === "all" ? "jasonsosa" : account }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
      }
      const { jobId } = await res.json();
      console.log("[generate] job enqueued:", jobId);

      let settled = false;
      function settle(progress: string, clearDelay: number) {
        if (settled) return;
        settled = true;
        cleanupGenPolling();
        setGenProgress(progress);
        setRefreshKey((k) => k + 1);
        setGenerating(false);
        settleTimerRef.current = setTimeout(() => setGenProgress(""), clearDelay);
      }

      pollAbortRef.current = new AbortController();
      const signal = pollAbortRef.current.signal;
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/admin/job-status?id=${jobId}`, { signal });
          if (!statusRes.ok) return;
          const job = await statusRes.json();

          if (job.status === "processing") {
            setGenProgress("Processing...");
          } else if (job.status === "completed") {
            const result = job.result;
            const msg = result?.skipped ? `Skipped: ${result.reason}`
              : result?.generated ? `Generated for ${result.hour}`
              : "Done";
            settle(msg, 8000);
          } else if (job.status === "failed") {
            settle(`Failed: ${job.error || "Unknown error"}`, 15000);
          }
        } catch {
          // Network error during poll, keep trying (abort errors will stop the interval)
        }
      }, 2000);

      // Intermediate feedback: if still polling after 30s, reassure the user
      feedbackRef.current = setTimeout(() => {
        if (!settled) {
          setGenProgress("Job queued, processing may take several minutes. You can navigate away safely.");
        }
      }, 30000);

      // Safety timeout: stop polling after 10 minutes
      safetyRef.current = setTimeout(() => {
        if (!settled) {
          settle("Job may still be processing. Check back later.", 15000);
        }
      }, 600000);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown";
      setGenProgress(`Failed: ${msg}`);
      console.error("[generate] error:", msg);
      setRefreshKey((k) => k + 1);
      setGenerating(false);
      settleTimerRef.current = setTimeout(() => setGenProgress(""), 15000);
    }
  }, [generating, account]);

  useEffect(() => {
    return () => { cleanupGenPolling(); };
  }, []);

  return {
    generating,
    genProgress,
    refreshKey,
    handleGenerate,
  };
}

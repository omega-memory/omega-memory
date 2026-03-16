import { useCallback, useEffect, useRef } from "react";

/**
 * Visibility-aware polling hook.
 * - Pauses polling when the tab is hidden
 * - Immediately refetches when the tab becomes visible again
 * - Cleans up interval on unmount
 */
export function useSmartPoll(
  callback: () => void | Promise<void>,
  intervalMs: number,
  { enabled = true }: { enabled?: boolean } = {},
) {
  const savedCallback = useRef(callback);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Always keep callback ref current
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  const start = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      savedCallback.current();
    }, intervalMs);
  }, [intervalMs]);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      stop();
      return;
    }

    // Start polling
    start();

    // Visibility handler: pause when hidden, refetch + resume when visible
    function handleVisibility() {
      if (document.hidden) {
        stop();
      } else {
        savedCallback.current();
        start();
      }
    }

    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [enabled, start, stop]);
}

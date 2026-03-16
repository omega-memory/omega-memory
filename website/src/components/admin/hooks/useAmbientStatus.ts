import { useState, useEffect, useCallback, useRef } from "react";

export interface AmbientStatus {
  activeAgents: number;
  totalSessions: number;
  recentMemories: number;
  fileConflicts: string[];
  sparkline: number[];
}

interface AmbientEvent {
  type: "agent_joined" | "agent_left" | "file_conflict" | "memory_spike";
  message: string;
  timestamp: number;
}

const EMPTY: AmbientStatus = {
  activeAgents: 0,
  totalSessions: 0,
  recentMemories: 0,
  fileConflicts: [],
  sparkline: [],
};

/**
 * Shared hook for ambient OMEGA status.
 * Polls /api/admin/ambient-status at the given interval (default 30s).
 * Emits events when significant state changes occur (for toasts).
 */
export function useAmbientStatus(pollInterval = 30_000) {
  const [status, setStatus] = useState<AmbientStatus>(EMPTY);
  const [events, setEvents] = useState<AmbientEvent[]>([]);
  const prevRef = useRef<AmbientStatus>(EMPTY);
  const initialRef = useRef(true);

  const dismissEvent = useCallback((timestamp: number) => {
    setEvents((prev) => prev.filter((e) => e.timestamp !== timestamp));
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/ambient-status");
      if (!res.ok) return;
      const data: AmbientStatus = await res.json();
      setStatus(data);

      // Detect changes and emit events (skip on first load)
      if (!initialRef.current) {
        const prev = prevRef.current;
        const now = Date.now();
        const newEvents: AmbientEvent[] = [];

        if (data.activeAgents > prev.activeAgents) {
          const diff = data.activeAgents - prev.activeAgents;
          newEvents.push({
            type: "agent_joined",
            message: `${diff} agent${diff > 1 ? "s" : ""} became active`,
            timestamp: now,
          });
        } else if (data.activeAgents < prev.activeAgents && prev.activeAgents > 0) {
          const diff = prev.activeAgents - data.activeAgents;
          newEvents.push({
            type: "agent_left",
            message: `${diff} agent${diff > 1 ? "s" : ""} went idle`,
            timestamp: now + 1,
          });
        }

        if (data.fileConflicts.length > prev.fileConflicts.length) {
          const newConflicts = data.fileConflicts.filter(
            (f) => !prev.fileConflicts.includes(f)
          );
          newConflicts.forEach((file, idx) => {
            newEvents.push({
              type: "file_conflict",
              message: `File conflict: ${file.split("/").pop()}`,
              timestamp: now + 2 + idx,
            });
          });
        }

        if (data.recentMemories > prev.recentMemories + 10) {
          newEvents.push({
            type: "memory_spike",
            message: `${data.recentMemories - prev.recentMemories} memories stored in last hour`,
            timestamp: now + 3,
          });
        }

        if (newEvents.length > 0) {
          setEvents((prev) => [...prev, ...newEvents].slice(-5));
        }
      }

      initialRef.current = false;
      prevRef.current = data;
    } catch {
      // Silently fail: ambient status is non-critical
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, pollInterval);

    // Pause polling when tab is hidden, refetch on visibility
    function handleVisibility() {
      if (document.hidden) return;
      fetchStatus();
    }
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [fetchStatus, pollInterval]);

  return { status, events, dismissEvent, refetch: fetchStatus };
}

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  CoordinationSession, CoordinationFileClaim, CoordinationFileRead,
  CoordinationMessage, CoordinationHandoff, CoordinationIntent,
  CoordinationDecision, CoordinationTask, CoordinationGitEvent,
  CoordinationMetric,
} from "../lib/types";

interface UseCoordinationDataReturn {
  sessions: CoordinationSession[];
  fileClaims: CoordinationFileClaim[];
  fileReads: CoordinationFileRead[];
  messages: CoordinationMessage[];
  handoffs: CoordinationHandoff[];
  intents: CoordinationIntent[];
  decisions: CoordinationDecision[];
  tasks: CoordinationTask[];
  gitEvents: CoordinationGitEvent[];
  metrics: CoordinationMetric[];
  isLoading: boolean;
  isHistorical: boolean;
  error: string | null;
  refetch: () => void;
}

interface UseCoordinationDataOptions {
  pollInterval?: number;
  since?: string | null;
  until?: string | null;
}

export function useCoordinationData(opts: UseCoordinationDataOptions = {}): UseCoordinationDataReturn {
  const { pollInterval = 15000, since = null, until = null } = opts;
  const [sessions, setSessions] = useState<CoordinationSession[]>([]);
  const [fileClaims, setFileClaims] = useState<CoordinationFileClaim[]>([]);
  const [fileReads, setFileReads] = useState<CoordinationFileRead[]>([]);
  const [messages, setMessages] = useState<CoordinationMessage[]>([]);
  const [handoffs, setHandoffs] = useState<CoordinationHandoff[]>([]);
  const [intents, setIntents] = useState<CoordinationIntent[]>([]);
  const [decisions, setDecisions] = useState<CoordinationDecision[]>([]);
  const [tasks, setTasks] = useState<CoordinationTask[]>([]);
  const [gitEvents, setGitEvents] = useState<CoordinationGitEvent[]>([]);
  const [metrics, setMetrics] = useState<CoordinationMetric[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const isHistorical = !!(since || until);

  const fetchData = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;
    try {
      // Single consolidated request: sessions + file data + all phase-2 data
      const params = new URLSearchParams();
      params.set("include", "all");
      if (since) params.set("since", since);
      if (until) params.set("until", until);
      const url = `/api/admin/coordination?${params.toString()}`;

      const res = await fetch(url, { signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setSessions(data.sessions ?? []);
      setFileClaims(data.file_claims ?? []);
      setFileReads(data.file_reads ?? []);
      setMessages(data.messages ?? []);
      setHandoffs(data.handoffs ?? []);
      setIntents(data.intents ?? []);
      setDecisions(data.decisions ?? []);
      setTasks(data.tasks ?? []);
      setGitEvents(data.git_events ?? []);
      setMetrics(data.metrics ?? []);

      setError(null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Fetch failed");
    } finally {
      setIsLoading(false);
    }
  }, [since, until]);

  useEffect(() => {
    setIsLoading(true);
    fetchData();

    // Only poll in live mode
    let intervalId: ReturnType<typeof setInterval> | undefined;
    if (!isHistorical && pollInterval > 0) {
      intervalId = setInterval(fetchData, pollInterval);
    }
    return () => {
      abortRef.current?.abort();
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchData, pollInterval, isHistorical]);

  return {
    sessions, fileClaims, fileReads,
    messages, handoffs, intents, decisions, tasks, gitEvents, metrics,
    isLoading, isHistorical, error, refetch: fetchData,
  };
}

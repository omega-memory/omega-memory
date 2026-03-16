import { useState, useEffect, useCallback, useRef } from "react";

export interface GraphNode {
  id: string;
  content: string;
  event_type: string | null;
  project: string | null;
  priority: number;
  tags: string[] | null;
  created_at: string;
  updated_at: string | null;
  session_id: string | null;
  // Enrichment fields
  memory_type: string | null;
  access_count: number;
  access_rank: number;
  entity_id: string | null;
  is_superseded: boolean;
  confidence: number | null;
  observation: string | null;
  facts: string[] | null;
  feedback_score: number | null;
  category: string | null;
}

export type EdgeType = "session" | "tag" | "related" | "contains_fact" | "same_entity" | "temporal_cluster" | "evolution" | "contradicts" | "supersedes";

export interface GraphLink {
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
}

export interface ProjectInfo {
  label: string;
  count: number;
  rawValues: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  projects: ProjectInfo[];
}

export function useMemoryGraph(projectFilter?: string) {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchGraph = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (projectFilter) params.set("project", projectFilter);
      const res = await fetch(`/api/admin/memories/graph?${params}`, { signal: controller.signal });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Failed to load graph");
    } finally {
      setLoading(false);
    }
  }, [projectFilter]);

  useEffect(() => {
    fetchGraph();
    return () => { abortRef.current?.abort(); };
  }, [fetchGraph]);

  return { data, setData, loading, error, refetch: fetchGraph };
}

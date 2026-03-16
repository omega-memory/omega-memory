import { useState, useEffect, useCallback, useRef } from "react";

export interface EntityNode {
  id: string;
  name: string;
  entityType: string;
  jurisdiction: string | null;
  status: string;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface EntityLink {
  source: string;
  target: string;
  type: string;
  metadata: Record<string, unknown> | null;
}

export interface EntityTypeInfo {
  type: string;
  count: number;
}

export interface EntitiesGraphData {
  nodes: EntityNode[];
  links: EntityLink[];
  entityTypes: EntityTypeInfo[];
  truncated?: boolean;
}

export function useEntitiesGraph() {
  const [data, setData] = useState<EntitiesGraphData | null>(null);
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
      const res = await fetch("/api/admin/entities/graph", { signal: controller.signal });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: EntitiesGraphData = await res.json();
      setData(json);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Failed to load entities");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph();
    return () => { abortRef.current?.abort(); };
  }, [fetchGraph]);

  return { data, loading, error, refetch: fetchGraph };
}

import { useState, useEffect, useCallback, useRef } from "react";

export interface SkillNode {
  id: string;
  name: string;
  plugin: string;
  description: string;
  type: "rigid" | "flexible";
  usageCount: number;
  lastUsed: string | null;
}

export interface SkillLink {
  source: string;
  target: string;
  type: "invocation" | "co-occurrence";
  weight: number;
}

export interface PluginInfo {
  id: string;
  label: string;
  color: string;
  count: number;
}

export interface SkillsGraphData {
  nodes: SkillNode[];
  links: SkillLink[];
  plugins: PluginInfo[];
}

export function useSkillsGraph(pluginFilter?: string) {
  const [data, setData] = useState<SkillsGraphData | null>(null);
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
      if (pluginFilter) params.set("plugin", pluginFilter);
      const res = await fetch(`/api/admin/skills/graph?${params}`, { signal: controller.signal });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Failed to load graph");
    } finally {
      setLoading(false);
    }
  }, [pluginFilter]);

  useEffect(() => {
    fetchGraph();
    return () => { abortRef.current?.abort(); };
  }, [fetchGraph]);

  return { data, loading, error, refetch: fetchGraph };
}

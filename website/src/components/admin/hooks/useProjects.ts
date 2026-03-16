import { useState, useEffect, useCallback, useMemo } from "react";

/**
 * Client-side project record matching the DB schema.
 */
export interface ProjectRecord {
  id: string;
  name: string;
  category: string;
  description: string | null;
  launch_progress: number;
  launch_status: string;
  is_paused: boolean;
  version: string | null;
  shipping_info: string | null;
  suggestion: string | null;
  links: { label: string; href: string }[];
  session_aliases: string[];
  color: string | null;
  path: string | null;
  tier: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

// Module-level cache for fast tab switches
let _moduleCache: { projects: ProjectRecord[]; ts: number; userId?: string } | null = null;
const CACHE_TTL = 60_000;

function extractBasename(rawPath: string): string {
  const cleaned = rawPath
    .replace(/\/\.claude\/worktrees\/[^/]+/, "")
    .replace(/\/\.worktrees\/[^/]+/, "")
    .replace(/\/+$/, "");
  const segments = cleaned.split("/").filter(Boolean);
  return segments[segments.length - 1] ?? "";
}

export function useProjects(userId?: string) {
  const [projects, setProjects] = useState<ProjectRecord[]>(_moduleCache?.projects ?? []);
  const [loading, setLoading] = useState(!_moduleCache);

  const refetch = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/projects/registry");
      if (!res.ok) return;
      const data = await res.json();
      const fetched: ProjectRecord[] = data.projects ?? [];
      setProjects(fetched);
      _moduleCache = { projects: fetched, ts: Date.now(), userId };
    } catch {
      // Keep existing data on error
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    const cacheValid = _moduleCache
      && Date.now() - _moduleCache.ts < CACHE_TTL
      && _moduleCache.userId === userId;
    if (cacheValid) {
      setProjects(_moduleCache!.projects);
      setLoading(false);
      return;
    }
    refetch();
  }, [refetch, userId]);

  const activeProjects = useMemo(
    () => projects.filter((p) => p.launch_status !== "archived"),
    [projects],
  );

  const getProject = useCallback(
    (id: string) => projects.find((p) => p.id === id),
    [projects],
  );

  const resolvePathToProject = useCallback(
    (rawPath: string | null | undefined): ProjectRecord | undefined => {
      if (!rawPath) return undefined;
      const cleaned = rawPath
        .replace(/\/\.claude\/worktrees\/[^/]+/, "")
        .replace(/\/\.worktrees\/[^/]+/, "")
        .replace(/\/+$/, "");

      // 1. Direct path match: session path starts with a project's registered path
      //    (longest match wins to handle nested projects like omega vs omega/website)
      let bestPathMatch: ProjectRecord | undefined;
      let bestPathLen = 0;
      for (const p of projects) {
        if (p.path && cleaned.startsWith(p.path) && p.path.length > bestPathLen) {
          bestPathMatch = p;
          bestPathLen = p.path.length;
        }
      }
      if (bestPathMatch) return bestPathMatch;

      // 2. Alias match: walk up path segments, first match wins
      const segments = cleaned.split("/").filter(Boolean);
      for (let i = segments.length - 1; i >= 0; i--) {
        const seg = segments[i];
        const match = projects.find((p) =>
          p.session_aliases.some((alias) => seg === alias),
        );
        if (match) return match;
      }
      return undefined;
    },
    [projects],
  );

  return {
    projects,
    activeProjects,
    loading,
    getProject,
    resolvePathToProject,
    refetch,
  };
}

import React, { useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useMemoryGraph, type GraphNode } from "./useMemoryGraph";
import { MemoryDetailPanel } from "./MemoryDetailPanel";
import { GraphLegend } from "./GraphLegend";

const MemoryGraph3D = lazy(() => import("./MemoryGraph3D"));

export default function MemoryGraphPage() {
  const [selectedProject, setSelectedProject] = useState<{ label: string; rawValues: string[] } | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [search, setSearch] = useState("");

  const projectFilterParam = selectedProject?.rawValues.join(",");
  const { data, setData, loading, error } = useMemoryGraph(projectFilterParam || undefined);

  // Raw values set for fast node-level project matching in the 3D graph
  const highlightRawValues = useMemo(
    () => selectedProject ? new Set(selectedProject.rawValues) : null,
    [selectedProject],
  );

  // Search: compute matching IDs (pass to 3D for dimming, keep all nodes visible)
  const searchMatchIds = useMemo(() => {
    if (!data || !search.trim()) return null;
    const q = search.toLowerCase();
    return new Set(
      data.nodes.filter((n) => n.content.toLowerCase().includes(q)).map((n) => n.id),
    );
  }, [data, search]);

  const handleSelectNode = useCallback((node: GraphNode | null) => {
    setSelectedNode(node);
  }, []);

  const handleSaveNode = useCallback(
    (updated: GraphNode) => {
      setSelectedNode(updated);
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          nodes: prev.nodes.map((n) => (n.id === updated.id ? updated : n)),
        };
      });
    },
    [setData],
  );

  return (
    <div className="fixed inset-0 z-50 bg-canvas flex flex-col">
      {/* Header */}
      <div className="flex-none flex items-center gap-3 px-4 py-3 border-b border-edge bg-surface/60 backdrop-blur-sm">
        {/* Back link */}
        <a
          href="/admin"
          className="text-ink-tertiary hover:text-ink text-[13px] mr-2"
        >
          &larr; Admin
        </a>

        {/* Title */}
        <h1 className="text-[15px] font-semibold text-ink">Memory Graph</h1>

        {/* Divider */}
        <div className="w-px h-5 bg-edge" />

        {/* Project filter dropdown */}
        <select
          value={selectedProject?.label ?? ""}
          onChange={(e) => {
            const val = e.target.value;
            if (!val) {
              setSelectedProject(null);
            } else {
              const project = data?.projects.find((p) => p.label === val);
              if (project) setSelectedProject({ label: project.label, rawValues: project.rawValues });
            }
          }}
          className="px-2.5 py-1.5 rounded-lg text-[12px] font-medium bg-surface-elevated text-ink-secondary border border-edge-subtle hover:border-edge-default focus:outline-none focus:border-gold/40 appearance-none cursor-pointer pr-7"
          style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 8px center" }}
        >
          <option value="">All projects</option>
          {data?.projects.map((project) => (
            <option key={project.label} value={project.label}>
              {project.label} ({project.count})
            </option>
          ))}
        </select>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Search */}
        <input
          type="text"
          placeholder="Search memories..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-52 px-3 py-1.5 rounded-lg text-[12px] bg-surface-elevated border border-edge-subtle text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40"
        />

        {/* Node count */}
        {data && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gold/10 border border-gold/20">
            <span className="text-[18px] font-semibold text-gold tabular-nums leading-none">
              {searchMatchIds ? searchMatchIds.size : data.nodes.length}
            </span>
            <span className="text-[11px] text-gold/60">
              {searchMatchIds ? `/ ${data.nodes.length} ` : ""}memories
            </span>
          </div>
        )}
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-ink-tertiary">
              Loading memories...
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-type-error">{error}</div>
          </div>
        )}

        {data && !loading && (
          <Suspense fallback={null}>
            <MemoryGraph3D
              data={data}
              onSelectNode={handleSelectNode}
              highlightProject={highlightRawValues}
              searchMatchIds={searchMatchIds}
            />
          </Suspense>
        )}

        {/* Legend */}
        {data && !loading && <GraphLegend />}

        {/* Detail panel */}
        {selectedNode && (
          <MemoryDetailPanel
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
            onSave={handleSaveNode}
          />
        )}
      </div>
    </div>
  );
}

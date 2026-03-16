import React, { useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useEntitiesGraph, type EntityNode } from "./useEntitiesGraph";
import { EntityDetailPanel } from "./EntityDetailPanel";

const EntitiesGraph3D = lazy(() => import("./EntitiesGraph3D"));

export default function EntitiesGraphPage() {
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<EntityNode | null>(null);
  const [search, setSearch] = useState("");

  const { data, loading, error } = useEntitiesGraph();

  // Client-side filtering by search + type
  const filteredData = useMemo(() => {
    if (!data) return data;
    let nodes = data.nodes;
    let links = data.links;

    if (typeFilter) {
      nodes = nodes.filter((n) => n.entityType === typeFilter);
      const nodeIds = new Set(nodes.map((n) => n.id));
      links = links.filter(
        (l) => nodeIds.has(l.source as string) && nodeIds.has(l.target as string),
      );
    }

    if (search.trim()) {
      const q = search.toLowerCase();
      nodes = nodes.filter(
        (n) =>
          n.name.toLowerCase().includes(q) ||
          n.entityType.includes(q) ||
          (n.jurisdiction || "").toLowerCase().includes(q),
      );
      const nodeIds = new Set(nodes.map((n) => n.id));
      links = links.filter(
        (l) => nodeIds.has(l.source as string) && nodeIds.has(l.target as string),
      );
    }

    return { ...data, nodes, links };
  }, [data, typeFilter, search]);

  const handleSelectNode = useCallback((node: EntityNode | null) => {
    setSelectedNode(node);
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-canvas flex flex-col">
      {/* Header */}
      <div className="flex-none flex items-center gap-3 px-4 py-3 border-b border-edge bg-surface/60 backdrop-blur-sm">
        <a
          href="/admin"
          className="text-ink-tertiary hover:text-ink text-[13px] mr-2"
        >
          &larr; Admin
        </a>

        <h1 className="text-[15px] font-semibold text-ink">Entity Graph</h1>

        <div className="w-px h-5 bg-edge" />

        {/* Type filter chips */}
        {data?.entityTypes.map((et) => (
          <button
            key={et.type}
            onClick={() => setTypeFilter(typeFilter === et.type ? null : et.type)}
            className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors ${
              typeFilter === et.type
                ? "bg-gold/20 text-gold border border-gold/30"
                : "bg-surface-elevated text-ink-secondary border border-edge-subtle hover:border-edge-default"
            }`}
          >
            {et.type.replace(/_/g, " ")}
            <span className="ml-1 opacity-50">{et.count}</span>
          </button>
        ))}

        <div className="flex-1" />

        {/* Search */}
        <input
          type="text"
          placeholder="Search entities..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-52 px-3 py-1.5 rounded-lg text-[12px] bg-surface-elevated border border-edge-subtle text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40"
        />

        {/* Count */}
        {filteredData && (
          <span className="text-[11px] text-ink-tertiary tabular-nums">
            {filteredData.nodes.length} entities
            {data?.truncated && !typeFilter && !search.trim() && (
              <span className="text-ink-faint ml-1">(showing 500 of 500+)</span>
            )}
          </span>
        )}
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-ink-tertiary">Loading entities...</div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-type-error">{error}</div>
          </div>
        )}

        {filteredData && !loading && filteredData.nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-center space-y-2">
              <div className="text-[14px] text-ink-secondary">No entities found</div>
              <div className="text-[12px] text-ink-tertiary">
                Entities are extracted from your OMEGA sessions automatically.
              </div>
            </div>
          </div>
        )}

        {filteredData && !loading && filteredData.nodes.length > 0 && (
          <Suspense fallback={null}>
            <EntitiesGraph3D
              data={filteredData}
              onSelectNode={handleSelectNode}
              highlightType={typeFilter}
            />
          </Suspense>
        )}

        {/* Detail panel */}
        {selectedNode && data && (
          <EntityDetailPanel
            node={selectedNode}
            graphData={data}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>
    </div>
  );
}

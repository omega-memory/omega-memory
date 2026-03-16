import React, { useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useSkillsGraph, type SkillNode } from "./useSkillsGraph";
import { SkillDetailPanel } from "./SkillDetailPanel";

const SkillsGraph3D = lazy(() => import("./SkillsGraph3D"));

export default function SkillsGraphPage() {
  const [pluginFilter, setPluginFilter] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<SkillNode | null>(null);
  const [search, setSearch] = useState("");

  const { data, loading, error } = useSkillsGraph(pluginFilter || undefined);

  const filteredData = useMemo(() => {
    if (!data || !search.trim()) return data;
    const q = search.toLowerCase();
    const matchingIds = new Set(
      data.nodes
        .filter((n) => n.name.toLowerCase().includes(q) || n.description.toLowerCase().includes(q))
        .map((n) => n.id),
    );
    return {
      ...data,
      nodes: data.nodes.filter((n) => matchingIds.has(n.id)),
      links: data.links.filter(
        (l) =>
          matchingIds.has(l.source as string) &&
          matchingIds.has(l.target as string),
      ),
    };
  }, [data, search]);

  const handleSelectNode = useCallback((node: SkillNode | null) => {
    setSelectedNode(node);
  }, []);

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
        <h1 className="text-[15px] font-semibold text-ink">Skills Graph</h1>

        {/* Divider */}
        <div className="w-px h-5 bg-edge" />

        {/* Plugin filter dropdown */}
        <select
          value={pluginFilter ?? ""}
          onChange={(e) => setPluginFilter(e.target.value || null)}
          className="px-2.5 py-1.5 rounded-lg text-[12px] font-medium bg-surface-elevated text-ink-secondary border border-edge-subtle hover:border-edge-default focus:outline-none focus:border-gold/40 appearance-none cursor-pointer pr-7"
          style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 8px center" }}
        >
          <option value="">All plugins</option>
          {data?.plugins.map((plugin) => (
            <option key={plugin.id} value={plugin.id}>
              {plugin.label} ({plugin.count})
            </option>
          ))}
        </select>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Search */}
        <input
          type="text"
          placeholder="Search skills..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-52 px-3 py-1.5 rounded-lg text-[12px] bg-surface-elevated border border-edge-subtle text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40"
        />

        {/* Node count */}
        {filteredData && (
          <span className="text-[11px] text-ink-tertiary tabular-nums">
            {filteredData.nodes.length} skills
          </span>
        )}
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-ink-tertiary">
              Loading skills...
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-[13px] text-type-error">{error}</div>
          </div>
        )}

        {filteredData && !loading && (
          <Suspense fallback={null}>
            <SkillsGraph3D
              data={filteredData}
              onSelectNode={handleSelectNode}
              highlightPlugin={pluginFilter}
            />
          </Suspense>
        )}

        {/* Detail panel */}
        {selectedNode && filteredData && (
          <SkillDetailPanel
            node={selectedNode}
            data={filteredData}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>
    </div>
  );
}

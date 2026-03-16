import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import type { EntityListItem, EntityListData } from "../lib/types";
import { TYPE_STYLES, DEFAULT_TYPE_STYLE, STATUS_STYLES, formatRelType } from "./constants";
import SlideDrawer from "../SlideDrawer";
import { useKeyboardNav } from "../hooks/useKeyboardNav";

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

const GRAPH_CACHE_TTL = 60_000;

interface GraphNode { id: string; name: string; entityType: string; }
interface GraphLink { source: string; target: string; type: string; } // 60s

// ── Detail Panel (List View) ──────────────────────────────
// This is the detail panel used in the list/table view (EntitiesTab).
// It receives an EntityListItem and fetches relationships from the
// cached graph endpoint.
//
// For the 3D graph view's detail panel, see:
//   graph/EntityDetailPanel.tsx
// That component receives an EntityNode + full graphData directly
// and renders within the graph page's layout (absolute positioning).

function EntityDetail({
  entity,
  onClose,
  fetchGraphCached,
}: {
  entity: EntityListItem;
  onClose: () => void;
  fetchGraphCached: () => Promise<{ nodes: GraphNode[]; links: GraphLink[] }>;
}) {
  const style = TYPE_STYLES[entity.entityType] || DEFAULT_TYPE_STYLE;
  const statusColor = STATUS_STYLES[entity.status] || "text-ink-secondary";

  // Fetch relationships for this entity
  const [relationships, setRelationships] = useState<{ type: string; targetName: string; direction: "out" | "in" }[]>([]);
  const [relLoading, setRelLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchGraphCached();
        if (cancelled) return;

        const nodeMap = new Map<string, string>();
        for (const n of data.nodes || []) nodeMap.set(n.id, n.name);

        const rels: typeof relationships = [];
        for (const link of data.links || []) {
          if (link.source === entity.id) {
            rels.push({ type: link.type, targetName: nodeMap.get(link.target) || link.target, direction: "out" });
          } else if (link.target === entity.id) {
            rels.push({ type: link.type, targetName: nodeMap.get(link.source) || link.source, direction: "in" });
          }
        }
        setRelationships(rels);
      } catch (err) { console.warn("[EntityDetail] graph fetch failed:", err); } finally {
        if (!cancelled) setRelLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [entity.id, fetchGraphCached]);

  return (
    <SlideDrawer open={true} onClose={onClose} title={entity.name} width="sm">
      <div className="px-4 py-3 space-y-4">
          {/* Type + Status */}
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider ${style.bg} ${style.text}`}>
              {formatRelType(entity.entityType)}
            </span>
            <span className={`text-[12px] font-medium ${statusColor}`}>{entity.status}</span>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-3 gap-3">
            <div className="px-3 py-2 rounded-lg bg-surface">
              <div className="text-[10px] uppercase tracking-wider text-ink-faint">Memories</div>
              <div className="text-[16px] font-semibold text-ink tabular-nums">{entity.memoryCount}</div>
            </div>
            <div className="px-3 py-2 rounded-lg bg-surface">
              <div className="text-[10px] uppercase tracking-wider text-ink-faint">Relations</div>
              <div className="text-[16px] font-semibold text-ink tabular-nums">{entity.relationshipCount}</div>
            </div>
            <div className="px-3 py-2 rounded-lg bg-surface">
              <div className="text-[10px] uppercase tracking-wider text-ink-faint">Projects</div>
              <div className="text-[16px] font-semibold text-ink tabular-nums">{entity.projects.length}</div>
            </div>
          </div>

          {/* Projects */}
          {entity.projects.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">Projects</div>
              <div className="flex flex-wrap gap-1.5">
                {entity.projects.map((p) => (
                  <span key={p} className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gold/10 text-gold border border-gold/20">
                    {p}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Jurisdiction */}
          {entity.jurisdiction && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Jurisdiction</div>
              <div className="text-[13px] text-ink">{entity.jurisdiction}</div>
            </div>
          )}

          {/* Metadata */}
          {entity.metadata && Object.keys(entity.metadata).length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">Details</div>
              <div className="space-y-1.5">
                {Object.entries(entity.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-2">
                    <span className="text-[12px] text-ink-tertiary">{key.replace(/_/g, " ")}</span>
                    <span className="text-[12px] text-ink text-right truncate max-w-[200px]">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Relationships */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">
              Relationships {!relLoading && relationships.length > 0 && `(${relationships.length})`}
            </div>
            {relLoading ? (
              <div className="flex items-center gap-2 py-2">
                <div className="w-3 h-3 rounded-full border border-ink-faint/30 border-t-ink-faint animate-spin" />
                <span className="text-[11px] text-ink-faint">Loading...</span>
              </div>
            ) : relationships.length === 0 ? (
              <div className="text-[12px] text-ink-faint italic">No relationships</div>
            ) : (
              <div className="space-y-1">
                {relationships.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 text-[12px]">
                    {r.direction === "out" ? (
                      <>
                        <span className="text-gold font-medium">{formatRelType(r.type)}</span>
                        <span className="text-ink-faint">&rarr;</span>
                        <span className="text-ink truncate">{r.targetName}</span>
                      </>
                    ) : (
                      <>
                        <span className="text-ink truncate">{r.targetName}</span>
                        <span className="text-ink-faint">&rarr;</span>
                        <span className="text-gold font-medium">{formatRelType(r.type)}</span>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Timestamps */}
          <div className="pt-2 border-t border-edge-subtle">
            <div className="grid grid-cols-2 gap-3 text-[11px]">
              <div>
                <span className="text-ink-faint">Created</span>
                <div className="text-ink-secondary">{new Date(entity.createdAt).toLocaleDateString()}</div>
              </div>
              {entity.lastSeen && (
                <div>
                  <span className="text-ink-faint">Last seen</span>
                  <div className="text-ink-secondary">{timeAgo(entity.lastSeen)}</div>
                </div>
              )}
            </div>
          </div>

          {/* ID */}
          <div className="pt-2 border-t border-edge-subtle">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Entity ID</div>
            <div className="text-[10px] text-ink-tertiary font-mono break-all">{entity.id}</div>
          </div>

          {/* View in 3D Graph link */}
          <a
            href="/admin/entities/graph"
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-[12px] font-medium text-gold bg-gold/[0.06] border border-gold/[0.12] hover:bg-gold/[0.1] transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3H21m-3.75 3H21" />
            </svg>
            View in 3D Graph
          </a>
      </div>
    </SlideDrawer>
  );
}

// ── Main Tab ───────────────────────────────────────────────

export default function EntitiesTab() {
  const [data, setData] = useState<EntityListData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [projectFilter, setProjectFilter] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<EntityListItem | null>(null);
  const [sortBy, setSortBy] = useState<"memories" | "name" | "recent">("memories");
  const [displayLimit, setDisplayLimit] = useState(50);

  // Issue #3: Graph cache scoped to component instance via useRef
  const graphCacheRef = useRef<{ data: { nodes: GraphNode[]; links: GraphLink[] }; ts: number } | null>(null);

  const fetchGraphCached = useCallback(async (): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> => {
    const cached = graphCacheRef.current;
    if (cached && Date.now() - cached.ts < GRAPH_CACHE_TTL) {
      return cached.data;
    }
    const res = await fetch("/api/admin/entities/graph");
    if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
    const gData = await res.json();
    graphCacheRef.current = { data: gData, ts: Date.now() };
    return gData;
  }, []);


  // Issue #15: Force re-render every 60s so timeAgo stays fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 60_000);
    return () => clearInterval(id);
  }, []);

  // Issue #8: Reset display limit when filters/search/sort change
  useEffect(() => {
    setDisplayLimit(50);
  }, [typeFilter, projectFilter, search, sortBy]);

  const fetchData = useCallback(async () => {
    // Invalidate graph cache so the detail panel fetches fresh
    // relationship data after entity list changes.
    graphCacheRef.current = null;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      // Type filter is applied client-side so we can compute accurate
      // per-type counts from the project-filtered set (E-8 fix).
      if (projectFilter) params.set("project", projectFilter);
      params.set("limit", "2000");
      const res = await fetch(`/api/admin/entities?${params}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: EntityListData = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load entities");
    } finally {
      setLoading(false);
    }
  }, [projectFilter]);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Client-side filtering pipeline:
  // 1. data.entities = project-filtered from API
  // 2. searchFiltered = + search filter (used for type chip counts -- E-8 fix)
  // 3. filtered = + type filter + sort (final display set)

  // After project (API) + search: base set for type chip counts
  const searchFiltered = useMemo(() => {
    if (!data) return [];
    let items = data.entities;
    if (search) {
      const q = search.toLowerCase();
      items = items.filter((e) =>
        e.name.toLowerCase().includes(q) ||
        e.entityType.toLowerCase().includes(q) ||
        e.projects.some((p) => p.toLowerCase().includes(q)),
      );
    }
    return items;
  }, [data, search]);

  // Compute type chip counts from searchFiltered (before type filter -- E-8 fix)
  const computedEntityTypes = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of searchFiltered) {
      counts.set(e.entityType, (counts.get(e.entityType) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({ type, count }));
  }, [searchFiltered]);

  // Apply type filter + sort for final display
  const filtered = useMemo(() => {
    let items = searchFiltered;
    if (typeFilter) {
      items = items.filter((e) => e.entityType === typeFilter);
    }
    if (sortBy === "name") {
      items = [...items].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === "recent") {
      items = [...items].sort((a, b) => {
        if (!a.lastSeen && !b.lastSeen) return 0;
        if (!a.lastSeen) return 1;
        if (!b.lastSeen) return -1;
        return b.lastSeen.localeCompare(a.lastSeen);
      });
    }
    // default "memories" sort is already from the API
    return items;
  }, [searchFiltered, typeFilter, sortBy]);

  // ─── Keyboard Nav ─────────────────────────────────
  const displayedEntityIds = useMemo(
    () => filtered.slice(0, displayLimit).map((e) => e.id),
    [filtered, displayLimit],
  );

  const { focusedId: focusedEntityId } = useKeyboardNav({
    dataAttribute: "data-entity-id",
    ids: displayedEntityIds,
    onSelect: (id) => {
      const entity = filtered.find((e) => e.id === id);
      if (entity) setSelected(entity);
    },
    enabled: !selected, // disable when detail panel is open
  });

  if (error) {
    return (
      <div className="px-5 pt-6">
        <div className="p-4 rounded-xl bg-red-500/[0.06] border border-red-500/[0.12] text-[13px] text-red-400">
          Failed to load entities: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 pt-6 pb-8 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-[18px] font-semibold text-ink">Entities</h2>
          <p className="text-[13px] text-ink-tertiary mt-0.5">
            {data ? `${data.totalUserEntities ?? data.entities.length} entities` : "Loading..."}
            {data && filtered.length !== (data.totalUserEntities ?? data.entities.length) && ` (${filtered.length} shown)`}
          </p>
        </div>
        <a
          href="/admin/entities/graph"
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] font-medium text-gold bg-gold/[0.06] border border-gold/[0.12] hover:bg-gold/[0.1] transition-colors shrink-0"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3H21m-3.75 3H21" />
          </svg>
          3D Graph
        </a>
      </div>

      {/* Stats cards -- E-7 fix: all stats derived from `filtered` so they
          stay consistent when search/project/type filters are active */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="px-4 py-3 rounded-xl bg-surface border border-edge">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Total</div>
            <div className="text-[20px] font-semibold text-ink tabular-nums">{filtered.length}</div>
          </div>
          <div className="px-4 py-3 rounded-xl bg-surface border border-edge">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Types</div>
            <div className="text-[20px] font-semibold text-ink tabular-nums">
              {new Set(filtered.map((e) => e.entityType)).size}
            </div>
          </div>
          <div className="px-4 py-3 rounded-xl bg-surface border border-edge">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">With Memories</div>
            <div className="text-[20px] font-semibold text-ink tabular-nums">
              {filtered.filter((e) => e.memoryCount > 0).length}
            </div>
          </div>
          <div className="px-4 py-3 rounded-xl bg-surface border border-edge">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Projects</div>
            <div className="text-[20px] font-semibold text-ink tabular-nums">
              {new Set(filtered.flatMap((e) => e.projects)).size}
            </div>
          </div>
        </div>
      )}

      {/* Type filter chips -- E-8 fix: counts from computedEntityTypes
          (project + search filtered, before type filter) */}
      {data && computedEntityTypes.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setTypeFilter(null)}
            className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
              !typeFilter
                ? "bg-gold/15 text-gold border-gold/25"
                : "bg-surface text-ink-tertiary border-edge hover:text-ink-secondary hover:border-edge"
            }`}
          >
            All
          </button>
          {computedEntityTypes.map(({ type, count }) => {
            const style = TYPE_STYLES[type] || DEFAULT_TYPE_STYLE;
            const isActive = typeFilter === type;
            return (
              <button
                key={type}
                onClick={() => setTypeFilter(isActive ? null : type)}
                className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
                  isActive
                    ? `${style.bg} ${style.text} border-current/25`
                    : "bg-surface text-ink-tertiary border-edge hover:text-ink-secondary"
                }`}
              >
                {formatRelType(type)} ({count})
              </button>
            );
          })}
        </div>
      )}

      {/* Project filter chips */}
      {data && data.projects.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-ink-faint self-center mr-1">Project:</span>
          <button
            onClick={() => setProjectFilter(null)}
            className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
              !projectFilter
                ? "bg-gold/15 text-gold border-gold/25"
                : "bg-surface text-ink-tertiary border-edge hover:text-ink-secondary"
            }`}
          >
            All
          </button>
          {data.projects.map(({ project, count }) => (
            <button
              key={project}
              onClick={() => setProjectFilter(projectFilter === project ? null : project)}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
                projectFilter === project
                  ? "bg-gold/15 text-gold border-gold/25"
                  : "bg-surface text-ink-tertiary border-edge hover:text-ink-secondary"
              }`}
            >
              {project} ({count})
            </button>
          ))}
        </div>
      )}

      {/* Search + Sort */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search entities..."
            className="w-full pl-9 pr-3 py-2 rounded-lg text-[13px] bg-surface border border-edge text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 transition-colors"
          />
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
          className="px-3 py-2 rounded-lg text-[12px] bg-surface border border-edge text-ink-secondary focus:outline-none focus:border-gold/40 transition-colors"
        >
          <option value="memories">Most memories</option>
          <option value="name">Name A-Z</option>
          <option value="recent">Most recent</option>
        </select>
      </div>

      {/* Entity list */}
      {loading ? (
        <div className="space-y-2">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-16 rounded-xl skeleton" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-[14px] text-ink-tertiary">No entities found</div>
          <div className="text-[12px] text-ink-faint mt-1">
            {search ? "Try a different search term" : "Entities are auto-extracted from stored memories"}
          </div>
        </div>
      ) : (
        <>
          {/* Issue #8: Show count indicator */}
          <div className="text-[12px] text-ink-faint">
            Showing {Math.min(displayLimit, filtered.length)} of {filtered.length} entities
          </div>
          <div className="space-y-1">
            {filtered.slice(0, displayLimit).map((entity) => {
              const style = TYPE_STYLES[entity.entityType] || DEFAULT_TYPE_STYLE;
              return (
                <button
                  key={entity.id}
                  data-entity-id={entity.id}
                  onClick={() => setSelected(entity)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-surface border border-edge hover:border-gold/20 hover:bg-surface-hover transition-all text-left group ${
                    focusedEntityId === entity.id ? "ring-1 ring-gold/40 bg-gold/[0.03]" : ""
                  }`}
                >
                  {/* Type dot */}
                  <span className={`w-2 h-2 rounded-full ${style.dot} shrink-0`} />

                  {/* Name + type */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-medium text-ink truncate group-hover:text-gold transition-colors">
                        {entity.name}
                      </span>
                      <span className={`text-[10px] font-semibold uppercase tracking-wider ${style.text} shrink-0`}>
                        {formatRelType(entity.entityType)}
                      </span>
                    </div>
                    {entity.projects.length > 0 && (
                      <div className="flex items-center gap-1 mt-0.5">
                        {entity.projects.slice(0, 3).map((p) => (
                          <span key={p} className="text-[10px] text-ink-faint bg-surface-hover px-1.5 py-0.5 rounded">
                            {p}
                          </span>
                        ))}
                        {entity.projects.length > 3 && (
                          <span className="text-[10px] text-ink-faint">+{entity.projects.length - 3}</span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-4 shrink-0">
                    {entity.memoryCount > 0 && (
                      <div className="text-right">
                        <div className="text-[12px] font-medium text-ink tabular-nums">{entity.memoryCount}</div>
                        <div className="text-[9px] text-ink-faint uppercase">mem</div>
                      </div>
                    )}
                    {entity.relationshipCount > 0 && (
                      <div className="text-right">
                        <div className="text-[12px] font-medium text-ink tabular-nums">{entity.relationshipCount}</div>
                        <div className="text-[9px] text-ink-faint uppercase">rel</div>
                      </div>
                    )}
                    {entity.lastSeen && (
                      <div className="text-[11px] text-ink-faint w-16 text-right">
                        {timeAgo(entity.lastSeen)}
                      </div>
                    )}
                  </div>

                  {/* Chevron */}
                  <svg className="w-4 h-4 text-ink-faint group-hover:text-ink-tertiary transition-colors shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
                  </svg>
                </button>
              );
            })}
          </div>
          {/* Issue #8: Show more button */}
          {displayLimit < filtered.length && (
            <button
              onClick={() => setDisplayLimit((prev) => prev + 50)}
              className="w-full py-2.5 rounded-xl text-[13px] font-medium text-gold bg-gold/[0.06] border border-gold/[0.12] hover:bg-gold/[0.1] transition-colors"
            >
              Show more ({filtered.length - displayLimit} remaining)
            </button>
          )}
        </>
      )}

      {/* Detail panel */}
      {selected && (
        <EntityDetail entity={selected} onClose={() => setSelected(null)} fetchGraphCached={fetchGraphCached} />
      )}
    </div>
  );
}

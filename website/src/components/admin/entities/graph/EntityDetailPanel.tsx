import { useEffect, useMemo, useState, useCallback } from "react";
import type { EntityNode, EntityLink, EntitiesGraphData } from "./useEntitiesGraph";
import { TYPE_STYLES, DEFAULT_TYPE_STYLE, STATUS_STYLES, formatRelType } from "../constants";

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// Detail panel for the 3D graph view (/admin/entities/graph).
// Receives an EntityNode and the full EntitiesGraphData so it can
// resolve relationships without an extra fetch.
//
// For the list view's detail panel, see:
//   ../EntitiesTab.tsx  (EntityDetail component)
// That component uses EntityListItem and fetches relationships
// from a cached graph endpoint.
export function EntityDetailPanel({
  node,
  graphData,
  onClose,
}: {
  node: EntityNode;
  graphData: EntitiesGraphData;
  onClose: () => void;
}) {
  // Issue #4: Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Issue #16: Copy-to-clipboard state
  const [copied, setCopied] = useState(false);
  const handleCopyId = useCallback(() => {
    navigator.clipboard.writeText(node.id).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [node.id]);

  const style = TYPE_STYLES[node.entityType] || DEFAULT_TYPE_STYLE;

  const statusColor = STATUS_STYLES[node.status] || "text-ink-secondary";

  // Issue #13: Pre-build adjacency map for O(1) relationship lookup
  const nodeMap = useMemo(
    () => new Map(graphData.nodes.map((n) => [n.id, n])),
    [graphData.nodes],
  );

  const adjacency = useMemo(() => {
    const map = new Map<
      string,
      { incoming: EntityLink[]; outgoing: EntityLink[] }
    >();
    for (const link of graphData.links) {
      const sourceId =
        typeof link.source === "string"
          ? link.source
          : (link.source as any).id;
      const targetId =
        typeof link.target === "string"
          ? link.target
          : (link.target as any).id;

      let srcEntry = map.get(sourceId);
      if (!srcEntry) {
        srcEntry = { incoming: [], outgoing: [] };
        map.set(sourceId, srcEntry);
      }
      srcEntry.outgoing.push(link);

      let tgtEntry = map.get(targetId);
      if (!tgtEntry) {
        tgtEntry = { incoming: [], outgoing: [] };
        map.set(targetId, tgtEntry);
      }
      tgtEntry.incoming.push(link);
    }
    return map;
  }, [graphData.links]);

  const adj = adjacency.get(node.id);
  const outgoing = (adj?.outgoing ?? [])
    .map((link) => {
      const targetId =
        typeof link.target === "string"
          ? link.target
          : (link.target as any).id;
      const target = nodeMap.get(targetId);
      return target ? { link, target } : null;
    })
    .filter(Boolean) as { link: EntityLink; target: EntityNode }[];

  const incoming = (adj?.incoming ?? [])
    .map((link) => {
      const sourceId =
        typeof link.source === "string"
          ? link.source
          : (link.source as any).id;
      const source = nodeMap.get(sourceId);
      return source ? { link, source } : null;
    })
    .filter(Boolean) as { link: EntityLink; source: EntityNode }[];

  return (
    <>
      {/* Backdrop */}
      <div className="absolute inset-0 z-10" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 z-20 w-[340px] border-l border-edge bg-surface/80 backdrop-blur-xl animate-slide-in-right overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider ${style.bg} ${style.text}`}
            >
              {node.entityType.replace(/_/g, " ")}
            </span>
            <button
              onClick={onClose}
              className="text-ink-tertiary hover:text-ink text-[13px] p-1 -m-1"
            >
              Esc
            </button>
          </div>

          {/* Name */}
          <h3 className="text-[16px] font-semibold text-ink leading-tight">
            {node.name}
          </h3>

          {/* Status + dates */}
          <div className="flex items-center gap-2 text-[12px] text-ink-tertiary">
            <span className={statusColor}>{node.status}</span>
            <span className="text-ink-faint">|</span>
            <span>{formatDate(node.createdAt)}</span>
          </div>

          {/* Jurisdiction */}
          {node.jurisdiction && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
                Jurisdiction
              </div>
              <div className="text-[13px] text-ink">
                {node.jurisdiction}
              </div>
            </div>
          )}

          {/* Metadata */}
          {node.metadata && Object.keys(node.metadata).length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Details
              </div>
              <div className="space-y-1.5">
                {Object.entries(node.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-2">
                    <span className="text-[12px] text-ink-tertiary">{key.replace(/_/g, " ")}</span>
                    <span className="text-[12px] text-ink text-right truncate max-w-[180px]">
                      {String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Outgoing relationships */}
          {outgoing.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Outgoing ({outgoing.length})
              </div>
              <div className="space-y-1.5">
                {outgoing.map(({ link, target }, i) => (
                  <div
                    key={`${node.id}-${target.id}-${link.type}-${i}`}
                    className="flex items-center gap-2 text-[12px]"
                  >
                    <span className="text-gold font-medium">{formatRelType(link.type)}</span>
                    <span className="text-ink-faint">&rarr;</span>
                    <span className="text-ink truncate">{target.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Incoming relationships */}
          {incoming.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Incoming ({incoming.length})
              </div>
              <div className="space-y-1.5">
                {incoming.map(({ link, source }, i) => (
                  <div
                    key={`${source.id}-${node.id}-${link.type}-${i}`}
                    className="flex items-center gap-2 text-[12px]"
                  >
                    <span className="text-ink truncate">{source.name}</span>
                    <span className="text-ink-faint">&rarr;</span>
                    <span className="text-gold font-medium">{formatRelType(link.type)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No relationships */}
          {outgoing.length === 0 && incoming.length === 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <span className="text-[11px] text-ink-faint">No relationships</span>
            </div>
          )}

          {/* ID (for debugging) */}
          <div className="pt-2 border-t border-edge-subtle">
            <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
              Entity ID
            </div>
            <div className="flex items-start gap-1.5">
              <div className="text-[11px] text-ink-secondary font-mono break-all flex-1">
                {node.id}
              </div>
              <button
                onClick={handleCopyId}
                className="flex-none text-[10px] text-ink-tertiary hover:text-ink px-1.5 py-0.5 rounded border border-edge-subtle hover:border-edge-default transition-colors"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

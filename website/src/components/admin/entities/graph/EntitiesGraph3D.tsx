import { useRef, useCallback, useMemo, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import type { EntityNode, EntityLink, EntitiesGraphData } from "./useEntitiesGraph";
import { ENTITY_TYPE_COLORS, DEFAULT_NODE_COLOR } from "../constants";

// Relationship type colors (edge)
const RELATIONSHIP_COLORS: Record<string, string> = {
  parent_of: "rgba(212, 168, 67, 0.25)",
  subsidiary_of: "rgba(212, 168, 67, 0.20)",
  owned_by: "rgba(212, 168, 67, 0.20)",
  acquired_by: "rgba(240, 96, 96, 0.20)",
  partner_of: "rgba(94, 201, 160, 0.20)",
  investor_in: "rgba(176, 136, 232, 0.20)",
  operated_by: "rgba(120, 120, 160, 0.20)",
  uses: "rgba(107, 159, 255, 0.20)",
  works_on: "rgba(94, 201, 160, 0.15)",
  depends_on: "rgba(107, 159, 255, 0.25)",
  mentions: "rgba(255, 255, 255, 0.06)",
  created_by: "rgba(94, 201, 160, 0.20)",
};

function getNodeColor(node: EntityNode, highlight?: string | null): string {
  if (highlight && node.entityType !== highlight) return "#1a1a2e";
  return ENTITY_TYPE_COLORS[node.entityType] || DEFAULT_NODE_COLOR;
}

export default function EntitiesGraph3D({
  data,
  onSelectNode,
  highlightType,
}: {
  data: EntitiesGraphData;
  onSelectNode: (node: EntityNode | null) => void;
  highlightType?: string | null;
}) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

  // Bloom post-processing
  useEffect(() => {
    if (!fgRef.current) return;
    try {
      Promise.all([
        import("three"),
        import("three/examples/jsm/postprocessing/UnrealBloomPass.js"),
      ]).then(([THREE, { UnrealBloomPass }]) => {
        if (fgRef.current) {
          // @ts-expect-error — Vector2 exists at runtime but not in inferred dynamic import types
          const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
          const bloom = new UnrealBloomPass(resolution, 0.5, 0.4, 0);
          fgRef.current.postProcessingComposer().addPass(bloom);
        }
      }).catch((err) => {
        console.warn("[EntitiesGraph3D] Bloom post-processing unavailable:", err);
      });
    } catch (err) {
      console.warn("[EntitiesGraph3D] Bloom post-processing unavailable:", err);
    }
  }, []);

  // Custom clustering force by entity_type
  useEffect(() => {
    if (!fgRef.current) return;
    const fg = fgRef.current;

    const charge = fg.d3Force("charge");
    if (charge) charge.strength(-80);

    // Clear previous cluster force to prevent accumulation
    fg.d3Force("cluster", null);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fg.d3Force("cluster", (alpha: number) => {
      const centers = new Map<string, { x: number; y: number; z: number; count: number }>();
      for (const node of data.nodes as any[]) {
        const t = node.entityType || "_other";
        const c = centers.get(t) || { x: 0, y: 0, z: 0, count: 0 };
        c.x += node.x || 0;
        c.y += node.y || 0;
        c.z += node.z || 0;
        c.count++;
        centers.set(t, c);
      }
      for (const c of centers.values()) {
        c.x /= c.count;
        c.y /= c.count;
        c.z /= c.count;
      }
      for (const node of data.nodes as any[]) {
        const t = node.entityType || "_other";
        const c = centers.get(t);
        if (!c) continue;
        const k = alpha * 0.25;
        node.vx = (node.vx || 0) + (c.x - (node.x || 0)) * k;
        node.vy = (node.vy || 0) + (c.y - (node.y || 0)) * k;
        node.vz = (node.vz || 0) + (c.z - (node.z || 0)) * k;
      }
    });
  }, [data]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback((node: any) => {
    onSelectNode(node as EntityNode);
    if (!fgRef.current) return;

    const distance = 60;
    const mag = Math.hypot(node.x || 0, node.y || 0, node.z || 0);
    const newPos = mag > 0.01
      ? {
          x: (node.x || 0) * (1 + distance / mag),
          y: (node.y || 0) * (1 + distance / mag),
          z: (node.z || 0) * (1 + distance / mag),
        }
      : { x: 0, y: 0, z: distance };

    fgRef.current.cameraPosition(newPos, node, 2000);
  }, [onSelectNode]);

  const handleBackgroundClick = useCallback(() => {
    onSelectNode(null);
  }, [onSelectNode]);

  // Pre-compute relationship counts per node for sizing
  const relCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const link of data.links) {
      const src = typeof link.source === "string" ? link.source : (link.source as any).id;
      const tgt = typeof link.target === "string" ? link.target : (link.target as any).id;
      counts.set(src, (counts.get(src) || 0) + 1);
      counts.set(tgt, (counts.get(tgt) || 0) + 1);
    }
    return counts;
  }, [data.links]);

  const graphData = useMemo(() => ({
    nodes: data.nodes,
    links: data.links,
  }), [data]);

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={graphData}
      backgroundColor="#08090f"
      nodeLabel={(node: any) => {
        const n = node as EntityNode;
        const color = ENTITY_TYPE_COLORS[n.entityType] || DEFAULT_NODE_COLOR;
        const typeLabel = n.entityType.replace(/_/g, " ");
        return `<div style="background:rgba(15,16,25,0.9);padding:6px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.07);max-width:260px;font-size:12px;color:#e8e8f0;font-family:Outfit,system-ui,sans-serif">
          <div style="color:${color};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px">${typeLabel}</div>
          <div style="font-weight:500">${n.name}</div>
          ${n.jurisdiction ? `<div style="color:#888;font-size:10px;margin-top:2px">${n.jurisdiction}</div>` : ""}
        </div>`;
      }}
      nodeColor={(node: any) => getNodeColor(node as EntityNode, highlightType)}
      nodeOpacity={0.85}
      nodeVal={(node: any) => {
        const count = relCounts.get((node as EntityNode).id) || 0;
        // Base size 2, scale up with relationships: 2 + sqrt(count) * 2
        // Gives a range roughly from 2 (isolated) to ~8+ (highly connected)
        return 2 + Math.sqrt(count) * 2;
      }}
      nodeResolution={16}
      linkColor={(link: any) => RELATIONSHIP_COLORS[(link as EntityLink).type] || "rgba(255,255,255,0.06)"}
      linkWidth={0.5}
      linkOpacity={0.7}
      linkDirectionalArrowLength={3}
      linkDirectionalArrowRelPos={0.9}
      linkDirectionalArrowColor={(link: any) => RELATIONSHIP_COLORS[(link as EntityLink).type] || "rgba(255,255,255,0.06)"}
      onNodeClick={handleNodeClick}
      onBackgroundClick={handleBackgroundClick}
      warmupTicks={80}
      cooldownTime={3000}
      showNavInfo={false}
    />
  );
}

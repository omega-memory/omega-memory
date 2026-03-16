import { useRef, useCallback, useMemo, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import { PLUGIN_COLORS } from "./skillsManifest";
import type { SkillNode, SkillLink, SkillsGraphData } from "./useSkillsGraph";

const DEFAULT_COLOR = "#505068";
const DIM_COLOR = "#1a1a2e";

function getNodeColor(node: SkillNode, highlightPlugin?: string | null): string {
  if (highlightPlugin && node.plugin !== highlightPlugin) return DIM_COLOR;
  return PLUGIN_COLORS[node.plugin] || DEFAULT_COLOR;
}

function getNodeSize(node: SkillNode): number {
  return 2 + Math.min(node.usageCount, 50) * 0.15;
}

export default function SkillsGraph3D({
  data,
  onSelectNode,
  highlightPlugin,
}: {
  data: SkillsGraphData;
  onSelectNode: (node: SkillNode | null) => void;
  highlightPlugin?: string | null;
}) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

  // Try to add bloom post-processing on mount
  useEffect(() => {
    if (!fgRef.current) return;
    try {
      import("three/examples/jsm/postprocessing/UnrealBloomPass.js").then(
        ({ UnrealBloomPass }) => {
          if (fgRef.current) {
            const bloom = new UnrealBloomPass(undefined, 0.5, 0.4, 0);
            fgRef.current.postProcessingComposer().addPass(bloom);
          }
        },
      ).catch(() => {
        // Bloom not available, graceful degradation
      });
    } catch {
      // Bloom not available
    }
  }, []);

  // Custom clustering force by plugin
  useEffect(() => {
    if (!fgRef.current) return;
    const fg = fgRef.current;

    const charge = fg.d3Force("charge");
    if (charge) charge.strength(-40);

    // Group nodes by plugin via a custom position force
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fg.d3Force("cluster", (alpha: number) => {
      const pluginCenters = new Map<string, { x: number; y: number; z: number; count: number }>();
      for (const node of data.nodes as any[]) {
        const p = node.plugin || "_none";
        const c = pluginCenters.get(p) || { x: 0, y: 0, z: 0, count: 0 };
        c.x += node.x || 0;
        c.y += node.y || 0;
        c.z += node.z || 0;
        c.count++;
        pluginCenters.set(p, c);
      }
      for (const c of pluginCenters.values()) {
        c.x /= c.count;
        c.y /= c.count;
        c.z /= c.count;
      }
      for (const node of data.nodes as any[]) {
        const p = node.plugin || "_none";
        const c = pluginCenters.get(p);
        if (!c) continue;
        const k = alpha * 0.3;
        node.vx = (node.vx || 0) + (c.x - (node.x || 0)) * k;
        node.vy = (node.vy || 0) + (c.y - (node.y || 0)) * k;
        node.vz = (node.vz || 0) + (c.z - (node.z || 0)) * k;
      }
    });
  }, [data]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback((node: any) => {
    onSelectNode(node as SkillNode);
    if (!fgRef.current) return;

    const distance = 60;
    const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);
    const newPos = (node.x || node.y || node.z)
      ? {
          x: (node.x || 0) * distRatio,
          y: (node.y || 0) * distRatio,
          z: (node.z || 0) * distRatio,
        }
      : { x: 0, y: 0, z: distance };

    fgRef.current.cameraPosition(newPos, node, 2000);
  }, [onSelectNode]);

  const handleBackgroundClick = useCallback(() => {
    onSelectNode(null);
  }, [onSelectNode]);

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
        const n = node as SkillNode;
        const color = PLUGIN_COLORS[n.plugin] || DEFAULT_COLOR;
        return `<div style="background:rgba(15,16,25,0.9);padding:6px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.07);max-width:260px;font-size:12px;color:#e8e8f0;font-family:Outfit,system-ui,sans-serif">
          <div style="color:${color};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px">${n.plugin} / ${n.type}</div>
          <div style="font-weight:600;margin-bottom:2px">${n.name}</div>
          <div style="color:#a0a0b8;font-size:11px">${n.description}</div>
          ${n.usageCount > 0 ? `<div style="color:#7878a0;font-size:10px;margin-top:3px">${n.usageCount} uses</div>` : ""}
        </div>`;
      }}
      nodeColor={(node: any) => getNodeColor(node as SkillNode, highlightPlugin)}
      nodeOpacity={0.85}
      nodeVal={(node: any) => getNodeSize(node as SkillNode)}
      nodeResolution={12}
      linkColor={(link: any) => {
        const l = link as SkillLink;
        return l.type === "invocation"
          ? "rgba(212, 168, 67, 0.4)"
          : "rgba(107, 159, 255, 0.15)";
      }}
      linkWidth={(link: any) => {
        const l = link as SkillLink;
        return l.type === "invocation" ? 1.2 : 0.3;
      }}
      linkDirectionalArrowLength={(link: any) => {
        const l = link as SkillLink;
        return l.type === "invocation" ? 4 : 0;
      }}
      linkDirectionalArrowRelPos={1}
      linkOpacity={0.6}
      onNodeClick={handleNodeClick}
      onBackgroundClick={handleBackgroundClick}
      warmupTicks={80}
      cooldownTime={3000}
      showNavInfo={false}
    />
  );
}

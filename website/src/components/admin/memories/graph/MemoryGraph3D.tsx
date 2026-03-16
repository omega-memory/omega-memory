import { useRef, useCallback, useMemo, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import type { GraphNode, GraphLink, GraphData } from "./useMemoryGraph";

const EVENT_TYPE_COLORS: Record<string, string> = {
  // High-weight (2.0-3.0): core knowledge
  constraint: "#ff9f43",
  checkpoint: "#ffc048",
  reminder: "#e8a040",
  decision: "#6b9fff",
  lesson_learned: "#5ec9a0",
  error_pattern: "#f06060",
  user_preference: "#b088e8",
  // Medium-weight (1.1-1.5): reflections & research
  task_completion: "#40c8c8",
  reflexion: "#78dbb8",
  outcome_evaluation: "#68c8a8",
  self_reflection: "#88e0c8",
  advisor_insight: "#a088e8",
  advisor_action_outcome: "#9078d8",
  sota_research: "#60b8e8",
  research_report: "#50a8d8",
  benchmark_update: "#70c0e0",
  sota_scan: "#80c8e0",
  preference_generated: "#c098f0",
  // Low-weight (0.05-1.0): coordination & ephemeral
  session_summary: "#7878a0",
  session_respawn: "#686890",
  coordination_snapshot: "#585878",
  file_conflict: "#a06868",
  merge_claim: "#887070",
  merge_release: "#887070",
  file_claimed: "#787088",
  file_released: "#787088",
  branch_claimed: "#787088",
  branch_released: "#787088",
  test: "#68a088",
  code_chunk: "#606878",
  file_summary: "#585868",
  // Say/Do contradiction tracking
  public_statement: "#e8c060",
  outcome_resolution: "#c8d860",
  contradiction_detected: "#f07878",
  entity_profile_update: "#d0a868",
};

const DEFAULT_COLOR = "#505068";

const LINK_TYPE_COLORS: Record<string, string> = {
  session: "rgba(212, 168, 67, 0.15)",
  tag: "rgba(107, 159, 255, 0.10)",
  contradicts: "rgba(240, 96, 96, 0.6)",
  supersedes: "rgba(240, 96, 96, 0.3)",
  evolution: "rgba(176, 136, 232, 0.5)",
  contains_fact: "rgba(94, 201, 160, 0.35)",
  related: "rgba(160, 160, 180, 0.15)",
  same_entity: "rgba(212, 168, 67, 0.5)",
  temporal_cluster: "rgba(140, 120, 180, 0.15)",
};

const LINK_TYPE_WIDTHS: Record<string, number> = {
  contradicts: 0.8,
  supersedes: 0.6,
  evolution: 0.5,
  contains_fact: 0.4,
  same_entity: 0.5,
  related: 0.3,
  temporal_cluster: 0.2,
  session: 0.2,
  tag: 0.2,
};

// Edge types that should show directional particles
const LINK_PARTICLE_COUNTS: Record<string, number> = {
  contradicts: 2,
  evolution: 1,
  supersedes: 1,
  contains_fact: 1,
};

// Shared geometry cache (created once, reused across all nodes)
const GLOW_GEOMETRY = new THREE.SphereGeometry(1, 16, 12);
const RING_GEOMETRY = new THREE.RingGeometry(1.2, 1.6, 24);
const OCTAHEDRON_GEOMETRY = new THREE.OctahedronGeometry(0.6, 0);
const TORUS_GEOMETRY = new THREE.TorusGeometry(0.5, 0.12, 8, 16);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ICOSAHEDRON_GEOMETRY = new (THREE as any).IcosahedronGeometry(0.5, 0);

// Binned glow materials (4 intensity levels to reduce unique materials)
const GLOW_MATERIALS = [0.25, 0.5, 0.75, 1.0].map(
  (intensity) =>
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(0xd4a843),
      transparent: true,
      opacity: 0.08 + intensity * 0.12,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
);

const SYSTEM_INSIGHT_GLOW_MATERIAL = new THREE.MeshBasicMaterial({
  color: new THREE.Color(0xd4a843),
  transparent: true,
  opacity: 0.18,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const SUPERSEDED_RING_MATERIAL = new THREE.MeshBasicMaterial({
  color: new THREE.Color(0xf06060),
  transparent: true,
  opacity: 0.6,
  side: THREE.DoubleSide,
  depthWrite: false,
});

const PROCEDURAL_MATERIAL = new THREE.MeshBasicMaterial({
  color: new THREE.Color(0x40c8c8),
  transparent: true,
  opacity: 0.5,
  wireframe: true,
});

const EPISODIC_MATERIAL = new THREE.MeshBasicMaterial({
  color: new THREE.Color(0xb088e8),
  transparent: true,
  opacity: 0.5,
  wireframe: true,
});

const SEMANTIC_MATERIAL = new THREE.MeshBasicMaterial({
  color: new THREE.Color(0x6b9fff),
  transparent: true,
  opacity: 0.5,
  wireframe: true,
});

// Gold/amber for system insight nodes (overrides default advisor_insight purple)
const SYSTEM_INSIGHT_COLOR = "#d4a843";

function getNodeColor(
  node: GraphNode,
  highlightProject?: Set<string> | null,
  searchMatchIds?: Set<string> | null,
): string {
  // System insights get gold/amber regardless of event_type
  const isSystemInsight = node.category === "system_insight";
  const base = isSystemInsight
    ? SYSTEM_INSIGHT_COLOR
    : (EVENT_TYPE_COLORS[node.event_type || ""] || DEFAULT_COLOR);
  // Dim nodes outside highlighted project group
  if (highlightProject && (!node.project || !highlightProject.has(node.project))) {
    return DEFAULT_COLOR;
  }
  // Dim nodes that don't match search
  if (searchMatchIds && !searchMatchIds.has(node.id)) {
    return DEFAULT_COLOR;
  }
  return base;
}

function getNodeSize(node: GraphNode): number {
  return 1 + (node.priority || 3) * 0.8;
}

export default function MemoryGraph3D({
  data,
  onSelectNode,
  highlightProject,
  searchMatchIds,
}: {
  data: GraphData;
  onSelectNode: (node: GraphNode | null) => void;
  highlightProject?: Set<string> | null;
  searchMatchIds?: Set<string> | null;
}) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

  // Try to add bloom post-processing on mount
  useEffect(() => {
    if (!fgRef.current) return;
    try {
      // Dynamic import to avoid SSR issues
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

  // Custom clustering force by project
  useEffect(() => {
    if (!fgRef.current) return;
    const fg = fgRef.current;

    const charge = fg.d3Force("charge");
    if (charge) charge.strength(-40);

    // Group nodes by project via a custom position force
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fg.d3Force("cluster", (alpha: number) => {
      const projectCenters = new Map<string, { x: number; y: number; z: number; count: number }>();
      for (const node of data.nodes as any[]) {
        const p = node.project || "_none";
        const c = projectCenters.get(p) || { x: 0, y: 0, z: 0, count: 0 };
        c.x += node.x || 0;
        c.y += node.y || 0;
        c.z += node.z || 0;
        c.count++;
        projectCenters.set(p, c);
      }
      for (const c of projectCenters.values()) {
        c.x /= c.count;
        c.y /= c.count;
        c.z /= c.count;
      }
      for (const node of data.nodes as any[]) {
        const p = node.project || "_none";
        const c = projectCenters.get(p);
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
    onSelectNode(node as GraphNode);
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

  const graphData = useMemo(() => {
    // Filter out links referencing nodes not in the dataset (e.g. entity nodes
    // returned separately by the API). Dangling links break d3-force simulation.
    const nodeIds = new Set(data.nodes.map((n) => n.id));
    return {
      nodes: data.nodes,
      links: data.links.filter((l) => nodeIds.has(l.source) && nodeIds.has(l.target)),
    };
  }, [data]);

  // Custom node overlays: glow, red ring, orbiting shapes
  // Returns null for nodes that need no overlay (fast default rendering)
  const nodeThreeObject = useCallback((node: unknown) => {
    const n = node as GraphNode;
    const hasGlow = n.access_rank > 0.5;
    const hasRing = n.is_superseded;
    const isProcedural = n.memory_type === "procedural";
    const isEpisodic = n.memory_type === "episodic";
    const isSemantic = n.memory_type === "semantic";
    const isSystemInsight = n.category === "system_insight";

    // Always return a group (never null) — returning null with nodeThreeObjectExtend
    // causes react-force-graph-3d v1.29.1 to skip default sphere rendering entirely
    const group = new THREE.Group();
    if (!hasGlow && !hasRing && !isProcedural && !isEpisodic && !isSemantic && !isSystemInsight) return group;
    const baseSize = getNodeSize(n);

    // Gold glow for system insight nodes
    if (isSystemInsight) {
      const glow = new THREE.Mesh(GLOW_GEOMETRY, SYSTEM_INSIGHT_GLOW_MATERIAL);
      const glowScale = baseSize * 2.0;
      glow.scale.set(glowScale, glowScale, glowScale);
      group.add(glow);
    }

    // Glow sphere for frequently accessed nodes
    if (hasGlow) {
      const bin = Math.min(3, Math.floor(n.access_rank * 4));
      const glow = new THREE.Mesh(GLOW_GEOMETRY, GLOW_MATERIALS[bin]);
      const glowScale = baseSize * (1.5 + n.access_rank * 2);
      glow.scale.set(glowScale, glowScale, glowScale);
      group.add(glow);
    }

    // Red ring for superseded memories
    if (hasRing) {
      const ring = new THREE.Mesh(RING_GEOMETRY, SUPERSEDED_RING_MATERIAL);
      const ringScale = baseSize * 0.8;
      ring.scale.set(ringScale, ringScale, ringScale);
      group.add(ring);
    }

    // Wireframe octahedron for procedural memory type
    if (isProcedural) {
      const octa = new THREE.Mesh(OCTAHEDRON_GEOMETRY, PROCEDURAL_MATERIAL);
      octa.position.set(baseSize * 1.8, 0, 0);
      group.add(octa);
    }

    // Wireframe torus for episodic memory type
    if (isEpisodic) {
      const torus = new THREE.Mesh(TORUS_GEOMETRY, EPISODIC_MATERIAL);
      torus.position.set(baseSize * 1.8, 0, 0);
      group.add(torus);
    }

    // Wireframe icosahedron for semantic memory type
    if (isSemantic) {
      const ico = new THREE.Mesh(ICOSAHEDRON_GEOMETRY, SEMANTIC_MATERIAL);
      ico.position.set(baseSize * 1.8, 0, 0);
      group.add(ico);
    }

    return group;
  }, []);

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={graphData}
      backgroundColor="#08090f"
      nodeLabel={(node: any) => {
        const n = node as GraphNode;
        const badges: string[] = [];
        badges.push(`<span style="color:${getNodeColor(n)};font-size:10px;text-transform:uppercase;letter-spacing:0.05em">${n.event_type || "memory"}</span>`);
        if (n.memory_type) {
          badges.push(`<span style="color:#888;font-size:10px;margin-left:6px">${n.memory_type}</span>`);
        }
        if (n.category === "system_insight") {
          badges.push(`<span style="color:#e63956;font-size:10px;margin-left:6px;font-weight:600">SYSTEM INSIGHT</span>`);
        }
        if (n.is_superseded) {
          badges.push(`<span style="color:#f06060;font-size:10px;margin-left:6px;font-weight:600">SUPERSEDED</span>`);
        }
        const metaLine: string[] = [];
        if (n.access_count > 0) metaLine.push(`${n.access_count} accesses`);
        if (n.confidence != null) metaLine.push(`${Math.round(n.confidence * 100)}% conf`);
        if (n.entity_id) metaLine.push(n.entity_id);

        return `<div style="background:rgba(15,16,25,0.92);padding:8px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.07);max-width:280px;font-size:12px;color:#e8e8f0;font-family:Outfit,system-ui,sans-serif">
          <div style="margin-bottom:3px;display:flex;align-items:center;flex-wrap:wrap">${badges.join("")}</div>
          <div>${n.content.slice(0, 120)}${n.content.length > 120 ? "..." : ""}</div>
          ${n.observation ? `<div style="color:#888;font-style:italic;margin-top:4px;font-size:11px">${n.observation.slice(0, 80)}${n.observation.length > 80 ? "..." : ""}</div>` : ""}
          ${metaLine.length > 0 ? `<div style="color:#666;font-size:10px;margin-top:4px">${metaLine.join(" · ")}</div>` : ""}
        </div>`;
      }}
      nodeColor={(node: any) => getNodeColor(node as GraphNode, highlightProject, searchMatchIds)}
      nodeOpacity={0.85}
      nodeVal={(node: any) => getNodeSize(node as GraphNode)}
      nodeResolution={12}
      linkColor={(link: any) => LINK_TYPE_COLORS[(link as GraphLink).type] || "rgba(255,255,255,0.05)"}
      linkWidth={(link: any) => LINK_TYPE_WIDTHS[(link as GraphLink).type] || 0.3}
      linkOpacity={0.6}
      linkDirectionalParticles={(link: any) => LINK_PARTICLE_COUNTS[(link as GraphLink).type] || 0}
      linkDirectionalParticleWidth={1.5}
      linkDirectionalParticleSpeed={0.006}
      linkDirectionalParticleColor={(link: any) => LINK_TYPE_COLORS[(link as GraphLink).type] || "rgba(255,255,255,0.1)"}
      nodeThreeObject={nodeThreeObject}
      nodeThreeObjectExtend={true}
      onNodeClick={handleNodeClick}
      onBackgroundClick={handleBackgroundClick}
      warmupTicks={80}
      cooldownTime={3000}
      showNavInfo={false}
    />
  );
}

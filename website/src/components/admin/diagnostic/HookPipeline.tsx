import { useState, useEffect, useCallback } from "react";
import type { HookPipelineData, HookNode } from "../lib/types";

const NODE_W = 140;
const NODE_H = 56;
const GAP_X = 40;
const GAP_Y = 24;
const PAD = 24;

const STATUS_COLORS = {
  success: { bg: "rgba(94,201,160,0.08)", border: "rgba(94,201,160,0.3)", text: "#5ec9a0", dot: "#5ec9a0" },
  error: { bg: "rgba(240,96,96,0.08)", border: "rgba(240,96,96,0.3)", text: "#f06060", dot: "#f06060" },
  inactive: { bg: "rgba(255,255,255,0.02)", border: "rgba(255,255,255,0.06)", text: "#666", dot: "#444" },
};

// Layout nodes in a horizontal line with pre_file_guard branching down
function layoutNodes(nodes: HookNode[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const mainLine = ["session_start", "pre_edit", "post_edit", "pre_push", "post_push", "session_stop"];
  const branch = ["pre_file_guard"];

  mainLine.forEach((id, i) => {
    positions.set(id, { x: PAD + i * (NODE_W + GAP_X), y: PAD });
  });

  branch.forEach((id) => {
    // Branch down from pre_edit
    const preEditPos = positions.get("pre_edit");
    if (preEditPos) {
      positions.set(id, { x: preEditPos.x, y: preEditPos.y + NODE_H + GAP_Y });
    }
  });

  // Add any nodes not in the predefined layout
  let nextX = PAD + mainLine.length * (NODE_W + GAP_X);
  for (const node of nodes) {
    if (!positions.has(node.id)) {
      positions.set(node.id, { x: nextX, y: PAD });
      nextX += NODE_W + GAP_X;
    }
  }

  return positions;
}

function NodeTooltip({ node }: { node: HookNode }) {
  return (
    <div className="absolute z-50 bottom-full mb-2 left-1/2 -translate-x-1/2 px-3 py-2 rounded-lg bg-[#1a1b20] border border-white/[0.08] shadow-lg shadow-black/30 whitespace-nowrap text-[12px]">
      <div className="font-medium text-ink mb-1">{node.label}</div>
      <div className="text-ink-secondary">Executions: <span className="text-ink tabular-nums">{node.executionCount}</span></div>
      <div className="text-ink-secondary">Avg: <span className="text-ink tabular-nums">{node.avgDurationMs}ms</span></div>
      {node.lastExecuted && (
        <div className="text-ink-faint mt-1">{new Date(node.lastExecuted).toLocaleString()}</div>
      )}
      {node.recentExecutions.length > 0 && (
        <div className="mt-2 space-y-0.5 border-t border-white/[0.06] pt-1.5">
          <div className="text-[10px] text-ink-faint uppercase tracking-wider">Recent</div>
          {node.recentExecutions.slice(0, 3).map((e, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${e.status === "error" ? "bg-type-error" : "bg-type-lesson"}`} />
              <span className="text-ink-secondary">{e.durationMs}ms</span>
              {e.error && <span className="text-type-error text-[11px] truncate max-w-[150px]">{e.error}</span>}
            </div>
          ))}
        </div>
      )}
      <span className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[5px] border-t-[#1a1b20]" />
    </div>
  );
}

export default function HookPipeline() {
  const [data, setData] = useState<HookPipelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/admin/hook-pipeline?days=7");
      if (res.ok) setData(await res.json());
    } catch {
      // Non-critical
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading && !data) {
    return (
      <section>
        <div className="admin-section-label">Hook Pipeline</div>
        <div className="admin-card h-40 skeleton rounded-xl" />
      </section>
    );
  }

  if (!data) return null;

  const positions = layoutNodes(data.nodes);
  const nodeMap = new Map(data.nodes.map(n => [n.id, n]));

  // Calculate SVG dimensions
  let maxX = 0, maxY = 0;
  for (const pos of positions.values()) {
    maxX = Math.max(maxX, pos.x + NODE_W);
    maxY = Math.max(maxY, pos.y + NODE_H);
  }
  const svgW = maxX + PAD;
  const svgH = maxY + PAD;

  return (
    <section>
      <div className="admin-section-label">Hook Pipeline (7d)</div>
      <div className="admin-card overflow-x-auto relative">
        <svg width={svgW} height={svgH} className="block">
          {/* Edges */}
          {data.edges.map(({ from, to }, i) => {
            const fromPos = positions.get(from);
            const toPos = positions.get(to);
            if (!fromPos || !toPos) return null;

            const x1 = fromPos.x + NODE_W;
            const y1 = fromPos.y + NODE_H / 2;
            const x2 = toPos.x;
            const y2 = toPos.y + NODE_H / 2;

            // If going downward (branch), use a curved path
            if (y2 > y1 + 10) {
              const midY = y1 + (y2 - y1) / 2;
              return (
                <path
                  key={i}
                  d={`M${x1},${y1} C${x1 + 20},${y1} ${x2 - 20},${midY} ${x2},${y2}`}
                  fill="none"
                  stroke="rgba(255,255,255,0.08)"
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                />
              );
            }

            return (
              <line
                key={i}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="rgba(255,255,255,0.08)"
                strokeWidth={1.5}
              />
            );
          })}

          {/* Arrowheads on edges */}
          {data.edges.map(({ to }, i) => {
            const toPos = positions.get(to);
            if (!toPos) return null;
            const x = toPos.x - 2;
            const y = toPos.y + NODE_H / 2;
            return (
              <polygon
                key={`arrow-${i}`}
                points={`${x},${y} ${x - 6},${y - 3} ${x - 6},${y + 3}`}
                fill="rgba(255,255,255,0.08)"
              />
            );
          })}

          {/* Nodes */}
          {data.nodes.map((node) => {
            const pos = positions.get(node.id);
            if (!pos) return null;
            const colors = STATUS_COLORS[node.status];
            const isHovered = hoveredNode === node.id;

            return (
              <g
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer"
              >
                <rect
                  x={pos.x} y={pos.y}
                  width={NODE_W} height={NODE_H}
                  rx={8}
                  fill={colors.bg}
                  stroke={isHovered ? colors.text : colors.border}
                  strokeWidth={isHovered ? 1.5 : 1}
                />
                {/* Status dot */}
                <circle
                  cx={pos.x + 14} cy={pos.y + NODE_H / 2}
                  r={4}
                  fill={colors.dot}
                />
                {/* Label */}
                <text
                  x={pos.x + 26} y={pos.y + 22}
                  fill={colors.text}
                  fontSize={11}
                  fontFamily="monospace"
                >
                  {node.label}
                </text>
                {/* Count */}
                <text
                  x={pos.x + 26} y={pos.y + 40}
                  fill="rgba(255,255,255,0.3)"
                  fontSize={10}
                  fontFamily="monospace"
                >
                  {node.executionCount > 0 ? `${node.executionCount}x / ${node.avgDurationMs}ms` : "inactive"}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Tooltip rendered as HTML overlay */}
        {hoveredNode && nodeMap.get(hoveredNode) && (
          <div
            className="absolute pointer-events-none"
            style={{
              left: (positions.get(hoveredNode)?.x ?? 0) + NODE_W / 2,
              top: (positions.get(hoveredNode)?.y ?? 0) - 8,
            }}
          >
            <NodeTooltip node={nodeMap.get(hoveredNode)!} />
          </div>
        )}
      </div>
    </section>
  );
}

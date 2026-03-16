import { useMemo, useId } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  type EdgeTypes,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";

import AgentNode, { type AgentNodeData } from "./nodes/AgentNode";
import FileNode from "./nodes/FileNode";
import InputFileNode from "./nodes/InputFileNode";
import ProjectGroupNode from "./nodes/ProjectGroupNode";
import type {
  CoordinationSession, CoordinationFileClaim, CoordinationFileRead,
  CoordinationMessage, CoordinationHandoff, CoordinationIntent,
  CoordinationDecision,
} from "../lib/types";
import { useProjects } from "../hooks/useProjects";
import { getAgentName, enrichProject, type Freshness, getFreshness } from "../lib/coordination-utils";

// ── Node types ───────────────────────────────────────────────────────

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  file: FileNode,
  inputFile: InputFileNode,
  projectGroup: ProjectGroupNode,
};

// ── Freshness colors ─────────────────────────────────────────────────

const FRESHNESS_COLORS = {
  active: { stroke: "rgba(52,211,153,0.45)", particle: "#34d399", glow: "0,211,153" },
  idle: { stroke: "rgba(251,191,36,0.35)", particle: "#fbbf24", glow: "251,191,36" },
  stale: { stroke: "rgba(248,113,113,0.25)", particle: "#f87171", glow: "248,113,113" },
} as const;

// Cyan colors for read edges (input side)
const READ_EDGE_COLORS = {
  stroke: "rgba(34,211,238,0.3)",
  particle: "#22d3ee",
  glow: "34,211,238",
};

// Freshness type imported from coordination-utils

// ── Message edge colors ──────────────────────────────────────────────

const MSG_TYPE_COLORS: Record<string, { stroke: string; particle: string; glow: string; label: string }> = {
  request:     { stroke: "rgba(251,191,36,0.4)",  particle: "#fbbf24", glow: "251,191,36",  label: "bg-amber-400/15 text-amber-400 border-amber-400/30" },
  inform:      { stroke: "rgba(96,165,250,0.4)",   particle: "#60a5fa", glow: "96,165,250",  label: "bg-blue-400/15 text-blue-400 border-blue-400/30" },
  acknowledge: { stroke: "rgba(52,211,153,0.4)",   particle: "#34d399", glow: "52,211,153",  label: "bg-emerald-400/15 text-emerald-400 border-emerald-400/30" },
  reject:      { stroke: "rgba(248,113,113,0.4)",  particle: "#f87171", glow: "248,113,113", label: "bg-red-400/15 text-red-400 border-red-400/30" },
  complete:    { stroke: "rgba(167,139,250,0.4)",   particle: "#a78bfa", glow: "167,139,250", label: "bg-violet-400/15 text-violet-400 border-violet-400/30" },
};

const DEFAULT_MSG_COLOR = MSG_TYPE_COLORS.inform;

// ── Particle edge ────────────────────────────────────────────────────

const PARTICLE_COUNT = 3;

function ParticleEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const reactId = useId();
  const glowId = `glow-${reactId}`;

  // Stable random durations per particle, keyed by edge id
  const particleDurations = useMemo(
    () => Array.from({ length: PARTICLE_COUNT }, () => 2.5 + Math.random() * 1.5),
    [id], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const edgeType = (data?.edgeType as "read" | "write") ?? "write";
  const freshness = (data?.freshness as Freshness) ?? "active";
  const colors = edgeType === "read" ? READ_EDGE_COLORS : FRESHNESS_COLORS[freshness];
  const task = data?.task as string | undefined;

  return (
    <>
      {/* SVG glow filter */}
      <defs>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Base path */}
      <path
        d={edgePath}
        fill="none"
        stroke={colors.stroke}
        strokeWidth={edgeType === "read" ? 1.5 : 2}
        className="transition-all duration-300"
      />

      {/* Flowing particles */}
      {Array.from({ length: PARTICLE_COUNT }, (_, i) => (
        <circle
          key={`${id}-particle-${i}`}
          r={edgeType === "read" ? 2 : 3}
          fill={colors.particle}
          filter={`url(#${glowId})`}
          opacity="0"
        >
          <animateMotion
            path={edgePath}
            dur={`${particleDurations[i]}s`}
            repeatCount="indefinite"
            begin={`${(i * 2.5) / PARTICLE_COUNT}s`}
          />
          <animate
            attributeName="opacity"
            values="0;0.9;0.9;0"
            keyTimes="0;0.1;0.9;1"
            dur={`${particleDurations[i]}s`}
            repeatCount="indefinite"
            begin={`${(i * 2.5) / PARTICLE_COUNT}s`}
          />
        </circle>
      ))}

      {/* Task label (write edges only) */}
      {task && edgeType !== "read" && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
            }}
            className="text-[9px] font-mono text-ink-faint bg-canvas/80 backdrop-blur-sm px-1.5 py-0.5 rounded-md border border-edge/40 max-w-[160px] truncate"
          >
            {task.length > 35 ? task.slice(0, 35) + "..." : task}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

// ── Message edge (dashed, agent-to-agent) ────────────────────────────

function MessageEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const reactId = useId();
  const glowId = `msg-glow-${reactId}`;

  // Stable random durations for message particles, keyed by edge id
  const particleDurations = useMemo(
    () => [4 + Math.random() * 2, 4 + Math.random() * 2],
    [id], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const isSelfLoop = sourceX === targetX && sourceY === targetY;

  // Self-loops (messages to/from non-visible sessions) get an arc above the node
  const [edgePath, labelX, labelY] = isSelfLoop
    ? [
        `M ${sourceX} ${sourceY - 20} C ${sourceX - 50} ${sourceY - 70}, ${sourceX + 50} ${sourceY - 70}, ${sourceX} ${sourceY - 20}`,
        sourceX,
        sourceY - 65,
      ]
    : getBezierPath({
        sourceX,
        sourceY,
        targetX,
        targetY,
        sourcePosition,
        targetPosition,
      });

  const msgType = (data?.msgType as string) ?? "inform";
  const colors = MSG_TYPE_COLORS[msgType] ?? DEFAULT_MSG_COLOR;
  const label = data?.label as string | undefined;
  const count = (data?.count as number) ?? 1;
  const hasUnread = (data?.hasUnread as boolean) ?? false;

  return (
    <>
      <defs>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={hasUnread ? 5 : 3} result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Dashed base path */}
      <path
        d={edgePath}
        fill="none"
        stroke={colors.stroke}
        strokeWidth={1.5}
        strokeDasharray="6 4"
        className="transition-all duration-300"
        opacity={hasUnread ? 1 : 0.8}
      />

      {/* 2 large slow particles */}
      {[0, 1].map((i) => (
        <circle
          key={`${id}-msg-particle-${i}`}
          r={4}
          fill={colors.particle}
          filter={`url(#${glowId})`}
          opacity="0"
        >
          <animateMotion
            path={edgePath}
            dur={`${particleDurations[i]}s`}
            repeatCount="indefinite"
            begin={`${i * 2.5}s`}
          />
          <animate
            attributeName="opacity"
            values="0;0.8;0.8;0"
            keyTimes="0;0.1;0.9;1"
            dur={`${particleDurations[i]}s`}
            repeatCount="indefinite"
            begin={`${i * 2.5}s`}
          />
        </circle>
      ))}

      {/* Label: msg_type badge + count */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "none",
          }}
          className="flex items-center gap-1"
        >
          <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${colors.label}`}>
            {msgType}
          </span>
          {count > 1 && (
            <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-surface-elevated border border-edge text-ink-secondary">
              {count}
            </span>
          )}
          {label && (
            <span className="text-[8px] text-ink-faint bg-canvas/80 px-1 py-0.5 rounded max-w-[100px] truncate">
              {label}
            </span>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

// ── Handoff edge (thick solid orange, agent-to-agent) ────────────────

function HandoffEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const reactId = useId();
  const glowId = `handoff-glow-${reactId}`;

  // Stable random duration for handoff particle, keyed by edge id
  const particleDuration = useMemo(
    () => 6 + Math.random() * 2,
    [id], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const context = data?.context as string | undefined;

  return (
    <>
      <defs>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Thick solid orange path */}
      <path
        d={edgePath}
        fill="none"
        stroke="rgba(251,146,60,0.5)"
        strokeWidth={3}
        className="transition-all duration-300"
      />

      {/* 1 large very slow particle */}
      <circle
        r={5}
        fill="#fb923c"
        filter={`url(#${glowId})`}
        opacity="0"
      >
        <animateMotion
          path={edgePath}
          dur={`${particleDuration}s`}
          repeatCount="indefinite"
          begin="0s"
        />
        <animate
          attributeName="opacity"
          values="0;0.9;0.9;0"
          keyTimes="0;0.1;0.9;1"
          dur={`${particleDuration}s`}
          repeatCount="indefinite"
          begin="0s"
        />
      </circle>

      {/* Label */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "none",
          }}
          className="flex items-center gap-1"
        >
          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded border bg-orange-400/15 text-orange-400 border-orange-400/30">
            HANDOFF
          </span>
          {context && (
            <span className="text-[8px] text-ink-faint bg-canvas/80 px-1 py-0.5 rounded max-w-[120px] truncate">
              {context}
            </span>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

// ── Intent overlap edge (violet dotted, bidirectional) ───────────────

function IntentOverlapEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const reactId = useId();
  const glowId = `intent-glow-${reactId}`;

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const overlapCount = (data?.overlapCount as number) ?? 0;
  const intentTypes = (data?.intentTypes as string[]) ?? [];
  const isRecent = (data?.isRecent as boolean) ?? false;

  return (
    <>
      <defs>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={isRecent ? 6 : 3} result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Dotted violet path */}
      <path
        d={edgePath}
        fill="none"
        stroke="rgba(167,139,250,0.4)"
        strokeWidth={1}
        strokeDasharray="3 3"
        filter={isRecent ? `url(#${glowId})` : undefined}
        className="transition-all duration-300"
      >
        {isRecent && (
          <animate
            attributeName="opacity"
            values="0.6;1;0.6"
            dur="2s"
            repeatCount="indefinite"
          />
        )}
      </path>

      {/* Label: overlap count + intent types */}
      {overlapCount > 0 && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
            }}
            className="flex items-center gap-1"
          >
            <span className="text-[8px] font-bold px-1.5 py-0.5 rounded border bg-violet-400/15 text-violet-400 border-violet-400/30">
              {overlapCount} file{overlapCount !== 1 ? "s" : ""}
            </span>
            {intentTypes.length > 0 && (
              <span className="text-[8px] text-violet-400/70 max-w-[80px] truncate">
                {intentTypes.join(", ")}
              </span>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

// ── Subagent edge (dashed gray bezier) ──────────────────────────────

function SubagentEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition }: EdgeProps) {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  return (
    <path
      id={id}
      d={edgePath}
      fill="none"
      stroke="rgba(148,163,184,0.35)"
      strokeWidth={1.5}
      strokeDasharray="4 3"
    />
  );
}

// ── Edge types registry ──────────────────────────────────────────────

const edgeTypes: EdgeTypes = {
  particle: ParticleEdge,
  message: MessageEdge,
  handoff: HandoffEdge,
  intentOverlap: IntentOverlapEdge,
  subagent: SubagentEdge,
};

// ── Layout helpers ───────────────────────────────────────────────────

const AGENT_WIDTH = 260;
const AGENT_HEIGHT = 100;
const FILE_WIDTH = 190;
const FILE_HEIGHT = 36;
const GROUP_LABEL_WIDTH = 200;
const GROUP_LABEL_HEIGHT = 32;
const GROUP_VERTICAL_GAP = 60;

function nodeSize(type: string | undefined) {
  if (type === "agent") return { w: AGENT_WIDTH, h: AGENT_HEIGHT };
  if (type === "projectGroup") return { w: GROUP_LABEL_WIDTH, h: GROUP_LABEL_HEIGHT };
  return { w: FILE_WIDTH, h: FILE_HEIGHT };
}

/** Layout a group of nodes + edges with dagre, offset by yOffset. */
function layoutGroup(
  nodes: Node[],
  edges: Edge[],
  yOffset: number,
): { laidOut: Node[]; height: number } {
  if (nodes.length === 0) return { laidOut: [], height: 0 };

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 30, ranksep: 140, marginx: 20, marginy: 20 });

  for (const node of nodes) {
    const { w, h } = nodeSize(node.type);
    g.setNode(node.id, { width: w, height: h });
  }

  for (const edge of edges) {
    if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
      g.setEdge(edge.source, edge.target);
    }
  }

  dagre.layout(g);

  let maxY = 0;
  const laidOut = nodes.map((node) => {
    const pos = g.node(node.id);
    const { w, h } = nodeSize(node.type);
    const y = pos.y - h / 2 + yOffset;
    maxY = Math.max(maxY, y + h);
    return {
      ...node,
      position: { x: pos.x - w / 2, y },
    };
  });

  return { laidOut, height: maxY - yOffset };
}

// ── Freshness helper (uses shared getFreshness from coordination-utils) ──

// ── Component ────────────────────────────────────────────────────────

type SortMode = "default" | "conflicts" | "uptime" | "files";

interface CoordinationFlowProps {
  sessions: CoordinationSession[];
  fileClaims: CoordinationFileClaim[];
  fileReads: CoordinationFileRead[];
  messages?: CoordinationMessage[];
  handoffs?: CoordinationHandoff[];
  intents?: CoordinationIntent[];
  decisions?: CoordinationDecision[];
  showMessages?: boolean;
  showHandoffs?: boolean;
  showIntents?: boolean;
  sortBy?: SortMode;
  playbackTime?: number | null;
  onNodeClick?: (type: "agent" | "file" | "inputFile", id: string) => void;
}

export default function CoordinationFlow({
  sessions,
  fileClaims,
  fileReads,
  messages = [],
  handoffs = [],
  intents = [],
  decisions = [],
  showMessages = true,
  showHandoffs = true,
  showIntents = true,
  sortBy = "default",
  playbackTime,
  onNodeClick,
}: CoordinationFlowProps) {
  const { resolvePathToProject } = useProjects();

  // Build a freshness lookup so edges can inherit the source agent's color
  const freshnessMap = useMemo(() => {
    const map: Record<string, Freshness> = {};
    const ref = playbackTime ?? undefined;
    for (const s of sessions) {
      map[s.session_id] = getFreshness(s.last_heartbeat, ref);
    }
    return map;
  }, [sessions, playbackTime]);

  // Set of visible session IDs for filtering communication edges
  const sessionIdSet = useMemo(
    () => new Set(sessions.map((s) => s.session_id)),
    [sessions],
  );

  // Detect file conflicts: same file_path claimed by multiple sessions
  const conflictPaths = useMemo(() => {
    const pathToSessions = new Map<string, string[]>();
    for (const f of fileClaims) {
      const existing = pathToSessions.get(f.file_path) ?? [];
      if (!existing.includes(f.session_id)) existing.push(f.session_id);
      pathToSessions.set(f.file_path, existing);
    }
    const conflicts = new Set<string>();
    for (const [path, sids] of pathToSessions) {
      if (sids.length > 1) conflicts.add(path);
    }
    return conflicts;
  }, [fileClaims]);

  // Filter out reads where the same session also claims the file (write supersedes read)
  const filteredReads = useMemo(() => {
    const claimedSet = new Set(
      fileClaims.map((c) => `${c.session_id}:${c.file_path}`),
    );
    return fileReads.filter(
      (r) => !claimedSet.has(`${r.session_id}:${r.file_path}`),
    );
  }, [fileReads, fileClaims]);

  // ── Build message edges ──────────────────────────────────────────
  const messageEdges = useMemo((): Edge[] => {
    if (!showMessages || messages.length === 0) return [];

    // Group by directed pair (from -> to), aggregate count and pick dominant type.
    // When only one endpoint is visible, anchor the edge as a self-loop on the
    // visible node so messages to/from ended sessions still appear.
    const pairMap = new Map<string, {
      from: string;
      to: string;
      count: number;
      msgType: string;
      subject: string;
      hasUnread: boolean;
    }>();

    for (const msg of messages) {
      const from = msg.from_session;
      const to = msg.to_session;
      if (!from || !to) continue; // Skip broadcasts (to_session=null)

      const fromVisible = sessionIdSet.has(from);
      const toVisible = sessionIdSet.has(to);
      if (!fromVisible && !toVisible) continue; // Neither endpoint visible

      // Resolve endpoints: if one side is missing, anchor to the visible node
      const resolvedFrom = fromVisible ? from : to;
      const resolvedTo = toVisible ? to : from;

      const key = `${resolvedFrom}->${resolvedTo}`;
      const existing = pairMap.get(key);
      if (existing) {
        existing.count++;
        if (!msg.read_at) existing.hasUnread = true;
      } else {
        pairMap.set(key, {
          from: resolvedFrom,
          to: resolvedTo,
          count: 1,
          msgType: msg.msg_type,
          subject: msg.subject,
          hasUnread: !msg.read_at,
        });
      }
    }

    return [...pairMap.entries()].map(([key, v]) => ({
      id: `msg-edge-${key}`,
      source: `agent-${v.from}`,
      target: `agent-${v.to}`,
      type: "message",
      data: {
        msgType: v.msgType,
        label: v.count === 1 ? (v.subject.length > 25 ? v.subject.slice(0, 25) + "..." : v.subject) : undefined,
        count: v.count,
        hasUnread: v.hasUnread,
      },
    }));
  }, [messages, showMessages, sessionIdSet]);

  // ── Build handoff edges ──────────────────────────────────────────
  const handoffEdges = useMemo((): Edge[] => {
    if (!showHandoffs || handoffs.length === 0) return [];

    const edges: Edge[] = [];
    for (const h of handoffs) {
      if (!sessionIdSet.has(h.session_id)) continue;

      // Parse read_by to find reader sessions among visible sessions
      let readers: string[] = [];
      try { readers = JSON.parse(h.read_by || "[]"); } catch { /* empty */ }

      for (const readerId of readers) {
        if (!sessionIdSet.has(readerId)) continue;
        edges.push({
          id: `handoff-edge-${h.id}-${readerId}`,
          source: `agent-${h.session_id}`,
          target: `agent-${readerId}`,
          type: "handoff",
          data: {
            context: h.key_context
              ? (h.key_context.length > 30 ? h.key_context.slice(0, 30) + "..." : h.key_context)
              : undefined,
          },
        });
      }
    }
    return edges;
  }, [handoffs, showHandoffs, sessionIdSet]);

  // ── Build intent overlap edges ───────────────────────────────────
  const intentOverlapEdges = useMemo((): Edge[] => {
    if (!showIntents || intents.length === 0) return [];

    // Only consider non-expired intents from visible sessions
    const now = Date.now();
    const activeIntents = intents.filter((i) => {
      if (!sessionIdSet.has(i.session_id)) return false;
      if (i.expires_at && new Date(i.expires_at).getTime() < now) return false;
      return true;
    });

    // Build map: file_path -> [{session_id, intent_type, created_at}]
    const fileToSessions = new Map<string, { sessionId: string; intentType: string; createdAt: string }[]>();
    for (const intent of activeIntents) {
      let files: string[] = [];
      try { files = JSON.parse(intent.target_files || "[]"); } catch { /* empty */ }
      for (const fp of files) {
        const list = fileToSessions.get(fp) ?? [];
        list.push({ sessionId: intent.session_id, intentType: intent.intent_type, createdAt: intent.created_at });
        fileToSessions.set(fp, list);
      }
    }

    // For each session pair sharing files, create one overlap edge
    const pairMap = new Map<string, { overlapCount: number; intentTypes: Set<string>; isRecent: boolean }>();
    for (const [, sessions] of fileToSessions) {
      const uniqueSessions = [...new Map(sessions.map((s) => [s.sessionId, s])).values()];
      if (uniqueSessions.length < 2) continue;

      for (let i = 0; i < uniqueSessions.length; i++) {
        for (let j = i + 1; j < uniqueSessions.length; j++) {
          const a = uniqueSessions[i];
          const b = uniqueSessions[j];
          const key = [a.sessionId, b.sessionId].sort().join("<->");
          const existing = pairMap.get(key);
          const recent = (now - new Date(a.createdAt).getTime()) < 5 * 60_000 ||
                         (now - new Date(b.createdAt).getTime()) < 5 * 60_000;
          if (existing) {
            existing.overlapCount++;
            existing.intentTypes.add(a.intentType);
            existing.intentTypes.add(b.intentType);
            if (recent) existing.isRecent = true;
          } else {
            pairMap.set(key, {
              overlapCount: 1,
              intentTypes: new Set([a.intentType, b.intentType]),
              isRecent: recent,
            });
          }
        }
      }
    }

    return [...pairMap.entries()].map(([key, v]) => {
      const [sA, sB] = key.split("<->");
      return {
        id: `intent-overlap-${key}`,
        source: `agent-${sA}`,
        target: `agent-${sB}`,
        type: "intentOverlap",
        data: {
          overlapCount: v.overlapCount,
          intentTypes: [...v.intentTypes],
          isRecent: v.isRecent,
        },
      };
    });
  }, [intents, showIntents, sessionIdSet]);

  // ── Build decision map for agent node data ───────────────────────
  const decisionsBySession = useMemo(() => {
    const map = new Map<string, { activeCount: number; domains: string[] }>();
    for (const d of decisions) {
      if (!sessionIdSet.has(d.decided_by)) continue;
      const existing = map.get(d.decided_by) ?? { activeCount: 0, domains: [] };
      if (d.status === "active") {
        existing.activeCount++;
        if (!existing.domains.includes(d.domain)) existing.domains.push(d.domain);
      }
      map.set(d.decided_by, existing);
    }
    return map;
  }, [decisions, sessionIdSet]);

  const { nodes, edges } = useMemo(() => {
    // Build per-session file paths for context on "Home" agents
    const sessionFiles = new Map<string, string[]>();
    for (const f of fileClaims) {
      const list = sessionFiles.get(f.session_id) ?? [];
      list.push(f.file_path);
      sessionFiles.set(f.session_id, list);
    }

    // Pre-pass: build parent-child maps from session metadata
    const parentChildMap = new Map<string, string>(); // child -> parent
    const childrenMap = new Map<string, string[]>();   // parent -> children[]
    for (const s of sessions) {
      const meta = s.metadata as Record<string, unknown> | null;
      const parentId = meta?.parent_session_id as string | undefined;
      if (parentId) {
        parentChildMap.set(s.session_id, parentId);
        const children = childrenMap.get(parentId) ?? [];
        if (!children.includes(s.session_id)) children.push(s.session_id);
        childrenMap.set(parentId, children);
      }
    }

    // Group sessions by project (enriched names)
    const projectGroups = new Map<string, CoordinationSession[]>();
    for (const s of sessions) {
      const { name } = enrichProject(s.project, sessionFiles.get(s.session_id), s.task, resolvePathToProject);
      const group = projectGroups.get(name) ?? [];
      group.push(s);
      projectGroups.set(name, group);
    }

    // Build write edges (agent -> file)
    const writeEdges: Edge[] = fileClaims.map((f, i) => ({
      id: `write-edge-${i}`,
      source: `agent-${f.session_id}`,
      target: `file-${f.session_id}-${f.file_path}`,
      type: "particle",
      data: {
        task: f.task,
        freshness: freshnessMap[f.session_id] ?? "active",
        edgeType: "write",
      },
    }));

    // Build read edges (inputFile -> agent)
    const readEdges: Edge[] = filteredReads.map((r, i) => ({
      id: `read-edge-${i}`,
      source: `input-${r.session_id}-${r.file_path}`,
      target: `agent-${r.session_id}`,
      type: "particle",
      data: {
        freshness: freshnessMap[r.session_id] ?? "active",
        edgeType: "read",
      },
    }));

    // Build subagent edges (parent -> child, dashed gray)
    const subagentEdges: Edge[] = [];
    for (const [childSid, parentSid] of parentChildMap) {
      subagentEdges.push({
        id: `subagent-edge-${parentSid}-${childSid}`,
        source: `agent-${parentSid}`,
        target: `agent-${childSid}`,
        type: "subagent",
        data: {},
      });
    }

    const allEdges = [...writeEdges, ...readEdges, ...messageEdges, ...handoffEdges, ...intentOverlapEdges, ...subagentEdges];

    // Layout each project group separately, stacked vertically
    const allNodes: Node[] = [];
    let yOffset = 0;
    let globalAnimIdx = 0;

    // Sort projects by selected criterion
    const sortedProjects = [...projectGroups.entries()].sort(([nameA, a], [nameB, b]) => {
      if (sortBy === "conflicts") {
        const aConflicts = a.reduce((n, s) => n + fileClaims.filter(f => f.session_id === s.session_id && conflictPaths.has(f.file_path)).length, 0);
        const bConflicts = b.reduce((n, s) => n + fileClaims.filter(f => f.session_id === s.session_id && conflictPaths.has(f.file_path)).length, 0);
        return bConflicts - aConflicts || nameA.localeCompare(nameB);
      }
      if (sortBy === "uptime") {
        const aMax = Math.max(...a.map(s => Date.now() - new Date(s.started_at).getTime()));
        const bMax = Math.max(...b.map(s => Date.now() - new Date(s.started_at).getTime()));
        return bMax - aMax;
      }
      if (sortBy === "files") {
        const aFiles = fileClaims.filter(f => a.some(s => s.session_id === f.session_id)).length;
        const bFiles = fileClaims.filter(f => b.some(s => s.session_id === f.session_id)).length;
        return bFiles - aFiles || nameA.localeCompare(nameB);
      }
      // default: active first, then alphabetical
      const aActive = a.some((s) => getFreshness(s.last_heartbeat, playbackTime ?? undefined) === "active");
      const bActive = b.some((s) => getFreshness(s.last_heartbeat, playbackTime ?? undefined) === "active");
      if (aActive !== bActive) return aActive ? -1 : 1;
      return nameA.localeCompare(nameB);
    });

    // Sort sessions within each project group by the same criterion
    for (const [, group] of sortedProjects) {
      group.sort((a, b) => {
        if (sortBy === "conflicts") {
          const ac = fileClaims.filter(f => f.session_id === a.session_id && conflictPaths.has(f.file_path)).length;
          const bc = fileClaims.filter(f => f.session_id === b.session_id && conflictPaths.has(f.file_path)).length;
          return bc - ac;
        }
        if (sortBy === "uptime") {
          return (Date.now() - new Date(b.started_at).getTime()) - (Date.now() - new Date(a.started_at).getTime());
        }
        if (sortBy === "files") {
          const af = fileClaims.filter(f => f.session_id === a.session_id).length;
          const bf = fileClaims.filter(f => f.session_id === b.session_id).length;
          return bf - af;
        }
        return 0;
      });
    }

    for (const [projName, projSessions] of sortedProjects) {
      const projSessionIds = new Set(projSessions.map((s) => s.session_id));
      const projClaims = fileClaims.filter((f) => projSessionIds.has(f.session_id));
      const projReads = filteredReads.filter((r) => projSessionIds.has(r.session_id));

      // Project label node
      const labelNode: Node = {
        id: `project-label-${projName}`,
        type: "projectGroup",
        position: { x: 0, y: 0 },
        data: {
          projectName: projName,
          agentCount: projSessions.length,
          fileCount: projClaims.length,
        },
      };

      // Agent nodes for this project
      const agentNodes: Node[] = projSessions.map((s) => {
        const idx = globalAnimIdx++;
        const dData = decisionsBySession.get(s.session_id);
        const parentSid = parentChildMap.get(s.session_id);
        return {
          id: `agent-${s.session_id}`,
          type: "agent",
          position: { x: 0, y: 0 },
          data: {
            sessionId: s.session_id,
            agentName: getAgentName(s.session_id),
            project: projName,
            status: s.status,
            task: s.task,
            lastHeartbeat: s.last_heartbeat,
            startedAt: s.started_at,
            freshness: getFreshness(s.last_heartbeat, playbackTime ?? undefined),
            claimedFiles: sessionFiles.get(s.session_id) ?? [],
            animIndex: idx,
            decisionCount: dData?.activeCount ?? 0,
            decisionDomains: dData?.domains.slice(0, 3) ?? [],
            isSubagent: !!parentSid,
            parentAgentName: parentSid ? getAgentName(parentSid) : undefined,
          } satisfies AgentNodeData,
        };
      });

      // Output file nodes (right column)
      const fileNodes: Node[] = projClaims.map((f, i) => ({
        id: `file-${f.session_id}-${f.file_path}`,
        type: "file",
        position: { x: 0, y: 0 },
        data: {
          filePath: f.file_path,
          sessionId: f.session_id,
          claimedAt: f.claimed_at,
          conflict: conflictPaths.has(f.file_path),
          animIndex: i,
        },
      }));

      // Input file nodes (left column)
      const inputFileNodes: Node[] = projReads.map((r, i) => ({
        id: `input-${r.session_id}-${r.file_path}`,
        type: "inputFile",
        position: { x: 0, y: 0 },
        data: {
          filePath: r.file_path,
          sessionId: r.session_id,
          firstReadAt: r.first_read_at,
          readCount: r.read_count,
          animIndex: i,
        },
      }));

      // Invisible edge from label to first agent for dagre grouping
      const groupEdges: Edge[] = agentNodes.length > 0
        ? [{ id: `group-edge-${projName}`, source: labelNode.id, target: agentNodes[0].id, type: "particle", hidden: true, data: {} }]
        : [];

      const groupNodes = [labelNode, ...inputFileNodes, ...agentNodes, ...fileNodes];
      const relevantEdges = [...groupEdges, ...allEdges.filter((e) =>
        groupNodes.some((n) => n.id === e.source) && groupNodes.some((n) => n.id === e.target),
      )];

      const { laidOut, height } = layoutGroup(groupNodes, relevantEdges, yOffset);
      allNodes.push(...laidOut);
      yOffset += height + GROUP_VERTICAL_GAP;
    }

    return { nodes: allNodes, edges: allEdges };
  }, [sessions, fileClaims, filteredReads, freshnessMap, conflictPaths, sortBy, playbackTime, messageEdges, handoffEdges, intentOverlapEdges, decisionsBySession]);

  return (
    <div className="w-full h-full bg-canvas overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={(_event, node) => {
          if (!onNodeClick) return;
          const type = node.type === "agent" ? "agent" : node.type === "inputFile" ? "inputFile" : "file";
          onNodeClick(type, node.id);
        }}
        fitView
        fitViewOptions={{ padding: 0.3, maxZoom: 1 }}
        minZoom={0.3}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{ type: "particle" }}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background gap={20} size={1} color="rgba(255,255,255,0.03)" />
        <Controls
          showInteractive={false}
          className="!bg-surface-elevated !border-edge !rounded-lg !shadow-md [&>button]:!bg-surface-elevated [&>button]:!border-edge [&>button]:!text-ink-secondary [&>button:hover]:!bg-surface"
        />
      </ReactFlow>
    </div>
  );
}

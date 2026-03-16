import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";
import { formatUptime } from "../../lib/coordination-utils";

export type AgentNodeData = {
  sessionId: string;
  agentName: string;
  project: string;
  status: string;
  task: string;
  lastHeartbeat: string;
  startedAt: string;
  freshness: "active" | "idle" | "stale";
  claimedFiles: string[];
  animIndex?: number;
  decisionCount?: number;
  decisionDomains?: string[];
  isSubagent?: boolean;
  parentAgentName?: string;
};

export type AgentNodeType = Node<AgentNodeData, "agent">;

// Glow colors per freshness state (used for box-shadow animation)
const GLOW = {
  active: { color: "52,211,153", dot: "bg-emerald-400" },    // emerald
  idle: { color: "251,191,36", dot: "bg-amber-400" },        // amber
  stale: { color: "248,113,113", dot: "bg-red-400" },        // red
} as const;

// Shimmer delay varies by index for organic stagger
const SHIMMER_DELAYS = [0, 1.2, 0.6, 1.8, 0.3];

/** Project name is now pre-resolved by CoordinationFlow via enrichProject. */
function displayProject(raw: string | null | undefined): string {
  if (!raw) return "Unknown project";
  return raw;
}

/** Infer a meaningful context from claimed file paths when project is "Home". */
function inferContext(files: string[]): string | null {
  if (files.length === 0) return null;
  // Find common directory segments across claimed files
  const dirs = files.map((f) => {
    const parts = f.split("/");
    // Find meaningful project-like segment (after "Projects/" or "src/")
    const projIdx = parts.findIndex((p) => p === "Projects" || p === "src");
    if (projIdx >= 0 && parts[projIdx + 1]) return parts[projIdx + 1];
    // Fallback: second-to-last directory
    return parts.length >= 2 ? parts[parts.length - 2] : parts[0];
  });
  // If all files share the same project context, use it
  const unique = [...new Set(dirs)];
  if (unique.length === 1) return unique[0];
  // Multiple contexts: show count
  return `${files.length} files`;
}

const FRESHNESS_LABEL = {
  active: "Active",
  idle: "Idle",
  stale: "Stale",
} as const;

function AgentNodeComponent({ data }: NodeProps<AgentNodeType>) {
  const { color, dot } = GLOW[data.freshness];
  const uptime = formatUptime(data.startedAt);
  const rawProject = displayProject(data.project);
  // For "Home" agents, try to infer a better label from claimed files
  const projectName =
    rawProject === "Home" && data.claimedFiles.length > 0
      ? inferContext(data.claimedFiles) ?? rawProject
      : rawProject;
  const idx = data.animIndex ?? 0;
  const isActive = data.freshness === "active";
  const isStale = data.freshness === "stale";
  const isSub = data.isSubagent ?? false;

  return (
    <motion.div
      initial={false}
      animate={{ opacity: isStale ? 0.5 : isSub ? 0.75 : 1 }}
      transition={{ duration: 0.3 }}
      className="relative"
      style={isSub ? { transform: "scale(0.9)", transformOrigin: "top left" } : undefined}
    >
      {/* Pulsing glow (active/idle only) */}
      {!isStale && (
        <motion.div
          className="absolute inset-0 rounded-xl"
          animate={{
            boxShadow: [
              `0 0 20px rgba(${color}, 0.15)`,
              `0 0 40px rgba(${color}, 0.3)`,
              `0 0 20px rgba(${color}, 0.15)`,
            ],
          }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
      )}

      <div
        className={`
          relative rounded-xl border bg-surface-elevated px-4 py-3
          ${isSub ? "min-w-[190px]" : "min-w-[220px]"} max-w-[280px] overflow-hidden
          transition-all duration-150
          hover:shadow-lg hover:shadow-black/10 hover:-translate-y-0.5
        `}
        style={{ borderColor: `rgba(${color}, 0.3)` }}
        title={`Agent: ${data.agentName} (${data.sessionId.slice(0, 8)})\nProject: ${projectName}\nTask: ${data.task || "none"}\nUptime: ${uptime}\nStatus: ${data.freshness}`}
      >
        <Handle type="target" position={Position.Left} className="!bg-cyan-400/60 !w-2 !h-2" />
        <Handle type="source" position={Position.Right} className="!bg-gold !w-2 !h-2" />

        {/* Shimmer sweep */}
        {!isStale && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.04] to-transparent"
            animate={{ x: ["-100%", "200%"] }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              delay: SHIMMER_DELAYS[idx % SHIMMER_DELAYS.length],
              repeatDelay: 3,
              ease: "easeInOut",
            }}
          />
        )}

        {/* Header: label + freshness + uptime */}
        <div className="relative flex items-center gap-1.5 mb-1">
          <span className="relative flex h-2 w-2 shrink-0">
            {isActive && (
              <span
                className={`absolute inset-0 rounded-full ${dot} animate-ping opacity-75`}
              />
            )}
            <span className={`relative inline-flex h-2 w-2 rounded-full ${dot}`} />
          </span>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint">
            {data.agentName} · {FRESHNESS_LABEL[data.freshness]}
          </span>
          {isSub && (
            <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-sky-400/10 text-sky-400 border border-sky-400/20 shrink-0">
              SUB
            </span>
          )}
          <span className="ml-auto text-[10px] text-ink-faint tabular-nums">
            {uptime}
          </span>
        </div>

        {/* Primary: project name */}
        <div className="relative text-[14px] font-semibold text-ink truncate">
          {projectName}
        </div>

        {/* Secondary: task description */}
        {data.task && (
          <div className="relative text-[11px] text-ink-secondary mt-0.5 truncate">
            {data.task}
          </div>
        )}

        {/* Subagent: spawned by indicator */}
        {isSub && data.parentAgentName && (
          <div className="relative text-[9px] text-sky-400/60 mt-0.5 truncate">
            spawned by {data.parentAgentName}
          </div>
        )}

        {/* Tertiary: session ID + decision badge */}
        <div className="relative flex items-center gap-1.5 mt-1">
          <span className="text-[9px] font-mono text-ink-faint truncate">
            {data.sessionId.slice(0, 8)}
          </span>
          {(data.decisionCount ?? 0) > 0 && (
            <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-emerald-400/10 text-emerald-400 border border-emerald-400/20 shrink-0">
              {data.decisionCount}D
            </span>
          )}
        </div>

        {/* Decision domain pills */}
        {data.decisionDomains && data.decisionDomains.length > 0 && (
          <div className="relative flex flex-wrap gap-1 mt-1">
            {data.decisionDomains.map((domain) => (
              <span
                key={domain}
                className="text-[8px] px-1.5 py-0.5 rounded-full bg-emerald-400/[0.07] text-emerald-400/70 border border-emerald-400/15 truncate max-w-[80px]"
              >
                {domain}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default memo(AgentNodeComponent);

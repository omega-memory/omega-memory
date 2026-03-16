import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";

export type FileNodeData = {
  filePath: string;
  sessionId: string;
  claimedAt: string;
  conflict?: boolean;
  animIndex?: number;
};

export type FileNodeType = Node<FileNodeData, "file">;

function truncatePath(p: string): string {
  const parts = p.split("/");
  if (parts.length <= 3) return p;
  return `.../${parts.slice(-2).join("/")}`;
}

function formatClaimAge(claimedAt: string): string {
  const ms = Date.now() - new Date(claimedAt).getTime();
  const min = Math.floor(ms / 60_000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  return `${h}h ago`;
}

function FileNodeComponent({ data }: NodeProps<FileNodeType>) {
  const idx = data.animIndex ?? 0;
  const isConflict = data.conflict ?? false;

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.3 + idx * 0.06, ease: "easeOut" }}
      className="relative"
    >
      {/* Conflict glow */}
      {isConflict && (
        <motion.div
          className="absolute inset-0 rounded-lg"
          animate={{
            boxShadow: [
              "0 0 8px rgba(248,113,113,0.3)",
              "0 0 20px rgba(248,113,113,0.5)",
              "0 0 8px rgba(248,113,113,0.3)",
            ],
          }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
      )}

      <div
        className={`
          relative rounded-lg border px-3 py-1.5 max-w-[200px]
          overflow-hidden transition-all duration-150
          hover:shadow-md hover:shadow-black/10 hover:-translate-y-0.5
          ${isConflict
            ? "border-red-400/50 bg-red-950/30 hover:border-red-400/70"
            : "border-edge bg-surface hover:border-gold/20"
          }
        `}
        title={`${data.filePath}\nClaimed by: ${data.sessionId.slice(0, 8)}\nClaimed: ${formatClaimAge(data.claimedAt)}${isConflict ? "\n⚠ CONFLICT: Multiple agents claim this file" : ""}`}
      >
        <Handle type="target" position={Position.Left} className="!bg-gold/60 !w-1.5 !h-1.5" />

        {/* Shimmer sweep */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.03] to-transparent"
          animate={{ x: ["-100%", "200%"] }}
          transition={{
            duration: 2,
            repeat: Infinity,
            delay: 1 + idx * 0.3,
            repeatDelay: 4,
            ease: "easeInOut",
          }}
        />

        <div className="relative flex items-center gap-1">
          {isConflict && (
            <span className="text-[10px] text-red-400 shrink-0" title="File claimed by multiple agents">
              ⚠
            </span>
          )}
          <span className={`text-[11px] font-mono truncate block ${isConflict ? "text-red-300" : "text-ink-tertiary"}`}>
            {truncatePath(data.filePath)}
          </span>
        </div>
        <span className="relative text-[9px] text-ink-faint block mt-0.5">
          {isConflict ? "Conflict" : formatClaimAge(data.claimedAt)}
        </span>
      </div>
    </motion.div>
  );
}

export default memo(FileNodeComponent);

import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";

export type InputFileNodeData = {
  filePath: string;
  sessionId: string;
  firstReadAt: string;
  readCount: number;
  animIndex?: number;
};

export type InputFileNodeType = Node<InputFileNodeData, "inputFile">;

function truncatePath(p: string): string {
  const parts = p.split("/");
  if (parts.length <= 3) return p;
  return `.../${parts.slice(-2).join("/")}`;
}

function formatReadAge(firstReadAt: string): string {
  const ms = Date.now() - new Date(firstReadAt).getTime();
  const min = Math.floor(ms / 60_000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  return `${h}h ago`;
}

function InputFileNodeComponent({ data }: NodeProps<InputFileNodeType>) {
  const idx = data.animIndex ?? 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.3 + idx * 0.06, ease: "easeOut" }}
      className="relative"
    >
      <div
        className={`
          relative rounded-lg border px-3 py-1.5 max-w-[200px]
          overflow-hidden transition-all duration-150
          border-cyan-500/30 bg-cyan-950/20
          hover:shadow-md hover:shadow-black/10 hover:-translate-y-0.5
          hover:border-cyan-400/50
        `}
        title={`${data.filePath}\nRead by: ${data.sessionId.slice(0, 8)}\nFirst read: ${formatReadAge(data.firstReadAt)}\nRead count: ${data.readCount}`}
      >
        <Handle type="source" position={Position.Right} className="!bg-cyan-400/60 !w-1.5 !h-1.5" />

        {/* Shimmer sweep */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/[0.03] to-transparent"
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
          <span className="text-[11px] font-mono truncate block text-cyan-300">
            {truncatePath(data.filePath)}
          </span>
          {data.readCount > 1 && (
            <span className="text-[8px] font-bold text-cyan-400/70 bg-cyan-400/10 rounded px-1 shrink-0">
              {data.readCount}x
            </span>
          )}
        </div>
        <span className="relative text-[9px] text-cyan-400/50 block mt-0.5">
          {formatReadAge(data.firstReadAt)}
        </span>
      </div>
    </motion.div>
  );
}

export default memo(InputFileNodeComponent);

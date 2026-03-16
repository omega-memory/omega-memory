import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type ProjectGroupNodeData = {
  projectName: string;
  agentCount: number;
  fileCount: number;
};

export type ProjectGroupNodeType = Node<ProjectGroupNodeData, "projectGroup">;

function ProjectGroupNodeComponent({ data }: NodeProps<ProjectGroupNodeType>) {
  return (
    <div className="px-3 py-1.5 pointer-events-none select-none">
      <Handle type="source" position={Position.Right} className="!opacity-0 !w-0 !h-0" />
      <div className="text-[13px] font-semibold uppercase tracking-wider text-ink-secondary">
        {data.projectName}
      </div>
      <div className="text-[10px] text-ink-faint mt-0.5">
        {data.agentCount} agent{data.agentCount !== 1 ? "s" : ""}
        {data.fileCount > 0 && ` · ${data.fileCount} file${data.fileCount !== 1 ? "s" : ""}`}
      </div>
    </div>
  );
}

export default memo(ProjectGroupNodeComponent);

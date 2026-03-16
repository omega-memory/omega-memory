import { useState } from "react";
import type { AttentionItem } from "./types";

interface NeedsAttentionBarProps {
  items: AttentionItem[];
  onClickItem: (projectId: string) => void;
}

export default function NeedsAttentionBar({
  items,
  onClickItem,
}: NeedsAttentionBarProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (items.length === 0) return null;

  return (
    <div className="mb-4 rounded-lg border border-amber-400/20 bg-amber-400/[0.04]">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm font-medium text-amber-400/80 hover:text-amber-400 transition-colors"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
          className={`transition-transform ${collapsed ? "-rotate-90" : ""}`}
        >
          <path d="M4 5.5L7 8.5L10 5.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>NEEDS ATTENTION ({items.length})</span>
      </button>

      {!collapsed && (
        <div className="px-4 pb-3 space-y-1.5">
          {items.map((item) => (
            <button
              key={item.projectId}
              onClick={() => onClickItem(item.projectId)}
              className="flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-sm hover:bg-white/[0.04] transition-colors"
            >
              <span
                className={`h-2 w-2 rounded-full flex-shrink-0 ${
                  item.severity === "red" ? "bg-red-400" : "bg-amber-400"
                }`}
              />
              <span className="text-white/70 font-medium">{item.projectName}</span>
              <span className="text-white/40">{item.reason}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

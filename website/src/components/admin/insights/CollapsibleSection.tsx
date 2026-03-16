import React, { useState } from "react";
import InfoTip from "../shared/InfoTip";

export default function CollapsibleSection({
  label,
  summary,
  tooltip,
  defaultOpen = false,
  children,
}: {
  label: string;
  summary?: string;
  tooltip?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="admin-card overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-3 w-full p-4 text-left touch-manipulation min-h-[44px]"
      >
        <h4 className="admin-section-label">{label}</h4>
        {tooltip && (
          <span onClick={(e) => e.stopPropagation()}>
            <InfoTip text={tooltip} />
          </span>
        )}
        {summary && !open && (
          <span className="text-[16px] text-ink-tertiary ml-1 truncate">{summary}</span>
        )}
        <svg
          width={14}
          height={14}
          viewBox="0 0 14 14"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`ml-auto text-ink-tertiary shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        >
          <path d="M3.5 5.25L7 8.75L10.5 5.25" />
        </svg>
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="px-4 pb-4">{children}</div>
        </div>
      </div>
    </div>
  );
}

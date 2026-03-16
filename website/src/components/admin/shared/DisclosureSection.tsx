import { useState, useEffect, useRef } from "react";

interface DisclosureSectionProps {
  title: string;
  defaultOpen?: boolean;
  badge?: string | number | null;
  storageKey?: string;
  children: React.ReactNode;
}

export default function DisclosureSection({
  title,
  defaultOpen = false,
  badge,
  storageKey,
  children,
}: DisclosureSectionProps) {
  const [isOpen, setIsOpen] = useState(() => {
    if (storageKey && typeof window !== "undefined") {
      const stored = localStorage.getItem(`omega-disclosure-${storageKey}`);
      if (stored !== null) return stored === "1";
    }
    return defaultOpen;
  });

  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (storageKey) {
      localStorage.setItem(`omega-disclosure-${storageKey}`, isOpen ? "1" : "0");
    }
  }, [isOpen, storageKey]);

  return (
    <div className="border border-edge rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-surface-hover transition-colors"
        aria-expanded={isOpen}
      >
        <svg
          className={`w-3.5 h-3.5 text-ink-faint transition-transform duration-200 ${isOpen ? "rotate-90" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
        <span className="text-[14px] font-semibold text-ink">{title}</span>
        {badge != null && (
          <span className="ml-auto text-[12px] text-ink-faint bg-surface-elevated px-2 py-0.5 rounded-full">
            {badge}
          </span>
        )}
      </button>
      <div
        ref={contentRef}
        className="grid transition-[grid-template-rows] duration-200 ease-out"
        style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="px-4 pb-4">{children}</div>
        </div>
      </div>
    </div>
  );
}

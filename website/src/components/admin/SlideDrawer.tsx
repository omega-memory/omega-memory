import { useEffect, useRef, useState } from "react";

type DrawerWidth = "sm" | "md" | "lg" | "xl";

const WIDTH_MAP: Record<DrawerWidth, string> = {
  sm: "max-w-[360px]",
  md: "max-w-[440px]",
  lg: "max-w-lg",
  xl: "max-w-[640px]",
};

interface DrawerTab {
  key: string;
  label: string;
}

interface SlideDrawerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  width?: DrawerWidth;
  tabs?: DrawerTab[];
  activeTab?: string;
  onTabChange?: (key: string) => void;
}

export default function SlideDrawer({ open, onClose, title, children, width = "lg", tabs, activeTab, onTabChange }: SlideDrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={title}>
      <div className="absolute inset-0 bg-canvas/60 backdrop-blur-sm" onClick={onClose} />
      <div ref={panelRef} className={`relative w-full ${WIDTH_MAP[width]} bg-surface border-l border-edge shadow-xl flex flex-col card-enter`}>
        <div className={`flex items-center justify-between px-5 py-4 border-b border-edge shrink-0 transition-shadow ${scrolled ? "shadow-[0_2px_8px_rgba(0,0,0,0.15)]" : ""}`}>
          <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center rounded-md text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover transition-colors" aria-label="Close">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        {tabs && tabs.length > 0 && (
          <div className="flex border-b border-edge-subtle px-5 shrink-0">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => onTabChange?.(tab.key)}
                className={`px-3 py-2.5 text-[13px] font-medium border-b-2 transition-colors ${
                  activeTab === tab.key
                    ? "border-gold text-gold"
                    : "border-transparent text-ink-tertiary hover:text-ink-secondary"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}
        <div
          className="flex-1 overflow-y-auto"
          onScroll={(e) => setScrolled((e.target as HTMLElement).scrollTop > 0)}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

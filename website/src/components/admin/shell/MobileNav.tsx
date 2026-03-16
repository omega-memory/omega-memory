import { type ReactElement, useEffect, useRef, useState } from "react";
import { type Tab, TAB_LABELS, TABS } from "../lib/types";

interface MobileNavProps {
  active: Tab;
  onNavigate: (tab: Tab) => void;
  actionsBadge: number;
}

const items: { id: Tab; label: string; iconStroke: ReactElement; iconFill: ReactElement }[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    iconStroke: (
      <svg className="w-[22px] h-[22px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25a2.25 2.25 0 0 1-2.25-2.25v-2.25Z" />
      </svg>
    ),
    iconFill: (
      <svg className="w-[22px] h-[22px]" viewBox="0 0 24 24" fill="currentColor">
        <path d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25a2.25 2.25 0 0 1-2.25-2.25v-2.25Z" />
      </svg>
    ),
  },
  {
    id: "feed",
    label: "Feed",
    iconStroke: (
      <svg className="w-[22px] h-[22px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
      </svg>
    ),
    iconFill: (
      <svg className="w-[22px] h-[22px]" viewBox="0 0 24 24" fill="currentColor">
        <path fillRule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25ZM12.75 6a.75.75 0 0 0-1.5 0v6c0 .414.336.75.75.75h4.5a.75.75 0 0 0 0-1.5h-3.75V6Z" clipRule="evenodd" />
      </svg>
    ),
  },
  {
    id: "actions",
    label: "Social",
    iconStroke: (
      <svg className="w-[22px] h-[22px]" fill="none" viewBox="0 0 22 22" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M11 1.5L3 5v5.5c0 4.7 3.4 9.1 8 10 4.6-.9 8-5.3 8-10V5l-8-3.5Z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 11l2.5 2.5 4.5-5" />
      </svg>
    ),
    iconFill: (
      <svg className="w-[22px] h-[22px]" viewBox="0 0 22 22" fill="currentColor">
        <path d="M11 1.5L3 5v5.5c0 4.7 3.4 9.1 8 10 4.6-.9 8-5.3 8-10V5l-8-3.5Zm3.8 7.3l-4.5 5a.75.75 0 0 1-1.1 0l-2.5-2.5a.75.75 0 1 1 1.1-1.1l2 1.9 4-4.4a.75.75 0 1 1 1.1 1.1Z" />
      </svg>
    ),
  },
  {
    id: "jobs",
    label: "Jobs",
    iconStroke: (
      <svg className="w-[22px] h-[22px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5" />
      </svg>
    ),
    iconFill: (
      <svg className="w-[22px] h-[22px]" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6.75 3a.75.75 0 0 1 .75.75V5.25h9V3.75a.75.75 0 0 1 1.5 0V5.25h1.5A2.25 2.25 0 0 1 21.75 7.5v11.25a2.25 2.25 0 0 1-2.25 2.25H4.5a2.25 2.25 0 0 1-2.25-2.25V7.5A2.25 2.25 0 0 1 4.5 5.25H6V3.75A.75.75 0 0 1 6.75 3ZM4.5 9v9.75c0 .414.336.75.75.75h13.5a.75.75 0 0 0 .75-.75V9H4.5Z" />
      </svg>
    ),
  },
];

/** Tabs not in the primary nav bar, shown in the overflow "More" menu */
const PRIMARY_IDS = new Set(items.map((i) => i.id));
const HIDDEN_TABS: Set<Tab> = new Set(["insights"]);
const overflowTabsAll: Tab[] = TABS.filter((t) => !PRIMARY_IDS.has(t) && !HIDDEN_TABS.has(t));

export default function MobileNav({ active, onNavigate, actionsBadge }: MobileNavProps) {
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);

  // Close overflow menu on outside click
  useEffect(() => {
    if (!moreOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) {
        setMoreOpen(false);
      }
    };
    window.addEventListener("mousedown", handleClick);
    return () => window.removeEventListener("mousedown", handleClick);
  }, [moreOpen]);

  // Close overflow menu on Escape
  useEffect(() => {
    if (!moreOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMoreOpen(false);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [moreOpen]);

  const overflowTabs = overflowTabsAll;
  const isActiveInOverflow = overflowTabs.includes(active);

  return (
    <nav
      className="lg:hidden fixed bottom-0 left-0 right-0 flex border-t border-edge-subtle bg-canvas/85 backdrop-blur-2xl safe-bottom z-20"
      role="tablist"
    >
      {items.map((tab) => {
        const isActive = active === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onNavigate(tab.id)}
            role="tab"
            aria-selected={isActive}
            aria-label={`${tab.label} tab`}
            className={`flex-1 flex flex-col items-center pt-2 pb-1.5 gap-0.5 transition-all duration-200 relative min-h-[52px] touch-manipulation ${
              isActive ? "text-ink" : "text-ink-faint hover:text-ink-tertiary"
            }`}
          >
            {isActive && (
              <span className="absolute top-0 left-1/2 -translate-x-1/2 w-10 h-[2px] rounded-full bg-gradient-to-r from-gold/60 via-gold to-gold/60 shadow-[0_0_8px_rgba(212,168,67,0.4)]" />
            )}
            <span className={`transition-transform duration-200 ${isActive ? "scale-105" : ""}`}>
              {isActive ? tab.iconFill : tab.iconStroke}
            </span>
            <span className={`text-[10px] tracking-wide transition-colors duration-200 ${
              isActive ? "font-semibold text-ink" : "font-medium text-ink-faint hidden sm:block"
            }`}>
              {tab.label}
            </span>
            {tab.id === "actions" && actionsBadge > 0 && (
              <span className="absolute top-1.5 left-1/2 translate-x-2 min-w-[16px] h-[16px] bg-type-error text-white text-[9px] font-bold rounded-full flex items-center justify-center px-1 shadow-[0_0_8px_rgba(240,96,96,0.4)]">
                {actionsBadge > 9 ? "9+" : actionsBadge}
              </span>
            )}
          </button>
        );
      })}

      {/* More overflow button */}
      <div ref={moreRef} className="relative flex-1 flex flex-col items-center">
        <button
          onClick={() => setMoreOpen(!moreOpen)}
          role="tab"
          aria-selected={isActiveInOverflow}
          aria-expanded={moreOpen}
          aria-label="More tabs"
          className={`flex-1 flex flex-col items-center pt-2 pb-1.5 gap-0.5 transition-all duration-200 relative min-h-[52px] touch-manipulation w-full ${
            isActiveInOverflow ? "text-ink" : "text-ink-faint hover:text-ink-tertiary"
          }`}
        >
          {isActiveInOverflow && (
            <span className="absolute top-0 left-1/2 -translate-x-1/2 w-10 h-[2px] rounded-full bg-gradient-to-r from-gold/60 via-gold to-gold/60 shadow-[0_0_8px_rgba(212,168,67,0.4)]" />
          )}
          <svg className="w-[22px] h-[22px]" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM12.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM18.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Z" />
          </svg>
          <span className={`text-[10px] tracking-wide transition-colors duration-200 ${
            isActiveInOverflow ? "font-semibold text-ink" : "font-medium text-ink-faint hidden sm:block"
          }`}>
            {isActiveInOverflow ? TAB_LABELS[active] : "More"}
          </span>
        </button>

        {/* Overflow menu */}
        {moreOpen && (
          <div role="menu" className="absolute bottom-full right-0 mb-2 w-48 bg-surface border border-edge rounded-xl shadow-[0_8px_32px_rgba(0,0,0,0.4)] overflow-hidden py-1 z-30">
            {overflowTabs.map((tabId) => {
              const isActive = active === tabId;
              return (
                <button
                  key={tabId}
                  role="menuitem"
                  onClick={() => {
                    onNavigate(tabId);
                    setMoreOpen(false);
                  }}
                  className={`w-full text-left px-4 py-2.5 text-[14px] transition-colors min-h-[44px] ${
                    isActive
                      ? "text-gold font-semibold bg-gold/[0.06]"
                      : "text-ink-secondary hover:bg-surface-elevated"
                  }`}
                >
                  {TAB_LABELS[tabId]}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
}

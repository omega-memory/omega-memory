interface HealthDot {
  label: string;
  status: "healthy" | "warning" | "error";
  detail: string;
}

interface TopBarProps {
  onSearchOpen: () => void;
  onMobileMenuOpen?: () => void;
  onLogout?: () => void;
  userEmail?: string | null;
  userRole?: "owner" | "contributor" | null;
  healthDots?: HealthDot[];
}

function StatusDot({ status }: { status: "healthy" | "warning" | "error" }) {
  if (status === "healthy") {
    return (
      <svg className="w-2.5 h-2.5" viewBox="0 0 10 10">
        <circle cx="5" cy="5" r="4" className="fill-emerald-400" filter="drop-shadow(0 0 3px rgba(52,211,153,0.5))" />
      </svg>
    );
  }
  if (status === "warning") {
    return (
      <svg className="w-2.5 h-2.5" viewBox="0 0 10 10">
        <path d="M5 1L9.33 8.5H0.67L5 1Z" className="fill-amber-400" filter="drop-shadow(0 0 3px rgba(251,191,36,0.5))" />
      </svg>
    );
  }
  return (
    <svg className="w-2.5 h-2.5" viewBox="0 0 10 10">
      <circle cx="5" cy="5" r="4" className="fill-none stroke-red-400 stroke-[1.5]" />
      <path d="M3.5 3.5L6.5 6.5M6.5 3.5L3.5 6.5" className="stroke-red-400 stroke-[1.5]" strokeLinecap="round" />
    </svg>
  );
}

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <span className="relative group/tip inline-flex" tabIndex={0}>
      {children}
      <span className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-full mt-2 px-3 py-1.5 rounded-lg bg-[#1a1b20] border border-white/[0.08] text-[12px] text-ink-secondary leading-snug whitespace-nowrap opacity-0 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100 transition-opacity duration-150 z-[100] shadow-lg shadow-black/30">
        {text}
        <span className="absolute left-1/2 -translate-x-1/2 bottom-full border-b-[5px] border-b-[#1a1b20] border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent" />
      </span>
    </span>
  );
}

export default function TopBar({ onSearchOpen, onMobileMenuOpen, onLogout, userEmail, userRole, healthDots }: TopBarProps) {
  return (
    <header className="topbar sticky top-0 z-20 flex items-center justify-between h-[52px] px-4 sm:px-6 bg-canvas/80 backdrop-blur-2xl border-b border-edge-subtle shrink-0 relative overflow-visible">
      {/* Mobile hamburger */}
      <div className="lg:hidden">
        <button
          onClick={onMobileMenuOpen}
          className="flex items-center justify-center w-9 h-9 rounded-md text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover transition-colors"
          aria-label="Open menu"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
          </svg>
        </button>
      </div>

      {/* Search trigger */}
      <button
        onClick={onSearchOpen}
        aria-label="Open search (Cmd+K)"
        className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.05] text-ink-faint/25 hover:text-ink-faint/40 hover:border-white/[0.08] transition-all max-w-[240px] w-full cursor-pointer"
      >
        <svg className="w-3.5 h-3.5 shrink-0 text-ink-faint/30" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
        </svg>
        <span className="text-[14px]">Search...</span>
        <kbd className="hidden sm:inline-flex ml-auto text-[10px] text-ink-faint/20 font-mono bg-white/[0.03] border border-white/[0.05] px-1.5 py-0.5 rounded">
          &#x2318;K
        </kbd>
      </button>

      <div className="flex-1" />

      {/* Health status dots */}
      {healthDots && healthDots.length > 0 && (
        <div className="hidden md:flex items-center gap-4" role="status" aria-label="System health status">
          {healthDots.map((h) => (
            <Tooltip key={h.label} text={h.detail}>
              <div className="flex items-center gap-2 cursor-default">
                <StatusDot status={h.status} />
                <span className="text-[13px] text-ink-faint/40">{h.label}</span>
              </div>
            </Tooltip>
          ))}
        </div>
      )}

      {healthDots && healthDots.length > 0 && (
        <div className="w-px h-6 bg-white/[0.06] hidden md:block mx-4" />
      )}

      {/* User info + Logout */}
      <div className="flex items-center gap-2">
        {userEmail && (
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="text-[14px] text-ink-tertiary/70 truncate max-w-[160px]">{userEmail}</span>
            {userRole && (
              <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-[10px] font-bold tracking-[0.1em] uppercase ${
                userRole === "owner"
                  ? "bg-gold/[0.06] text-gold/60 border border-gold/[0.08]"
                  : "bg-surface-elevated text-ink-faint border border-edge-subtle"
              }`}>
                {userRole}
              </span>
            )}
          </div>
        )}
        <button
          onClick={onLogout}
          className="w-8 h-8 flex items-center justify-center rounded-lg text-ink-faint/30 hover:text-ink-tertiary hover:bg-white/[0.04] transition-colors cursor-pointer"
          aria-label="Sign out"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15m3 0 3-3m0 0-3-3m3 3H9" />
          </svg>
        </button>
      </div>
    </header>
  );
}

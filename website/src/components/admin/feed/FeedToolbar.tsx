import { useState, useEffect } from "react";
import { PRIMARY_FILTERS, SECONDARY_FILTERS, type SortOption } from "./feedScoring";
import { PROJECT_NAMES } from "../contentUtils/types";

interface FeedToolbarProps {
  filter: string;
  onFilterChange: (filter: string) => void;
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
  density: "default" | "compact";
  onDensityChange: (d: "default" | "compact") => void;
  refreshing: boolean;
  onRefresh: () => void;
  onIngestUrl: (url: string) => Promise<{ success: boolean; domain?: string; error?: string }>;
  showArchived: boolean;
  onShowArchivedChange: (show: boolean) => void;
  selectedProject: string;
  onProjectChange: (project: string) => void;
  availableProjects: string[];
  viewMode: "timeline" | "summary";
  onViewModeChange: (mode: "timeline" | "summary") => void;
}

export default function FeedToolbar({
  filter,
  onFilterChange,
  sortBy,
  onSortChange,
  density,
  onDensityChange,
  refreshing,
  onRefresh,
  onIngestUrl,
  showArchived,
  onShowArchivedChange,
  selectedProject,
  onProjectChange,
  availableProjects,
  viewMode,
  onViewModeChange,
}: FeedToolbarProps) {
  const [showMore, setShowMore] = useState(false);
  const [showIngest, setShowIngest] = useState(false);
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestSuccess, setIngestSuccess] = useState<string | null>(null);
  const [showSortMenu, setShowSortMenu] = useState(false);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        if (showSortMenu) { setShowSortMenu(false); e.stopPropagation(); }
        if (showMore) { setShowMore(false); e.stopPropagation(); }
      }
    }
    if (showMore || showSortMenu) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [showMore, showSortMenu]);

  const primaryFilters = PRIMARY_FILTERS;
  const moreFilters = SECONDARY_FILTERS;
  const isMoreActive = moreFilters.some((f) => f.value === filter);

  const sortLabels: Record<SortOption, string> = {
    priority: "Priority",
    newest: "Newest",
    oldest: "Oldest",
    unread: "Unread First",
  };

  const humanizeProject = (slug: string) => PROJECT_NAMES[slug] || slug;

  return (
    <div className="px-5 pt-3 pb-1 space-y-2">
      {/* Project dropdown */}
      {availableProjects.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-[13px] text-ink-faint font-mono uppercase tracking-wider">Project</span>
          <select
            value={selectedProject}
            onChange={(e) => onProjectChange(e.target.value)}
            className="appearance-none bg-surface-elevated border border-edge rounded-lg px-2.5 py-1 text-[14px] text-ink font-medium cursor-pointer hover:border-edge-strong focus:border-gold/30 focus:outline-none transition-colors pr-7 bg-no-repeat bg-[length:12px] bg-[right_8px_center]"
            style={{ backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23666' stroke-width='2'%3e%3cpath stroke-linecap='round' stroke-linejoin='round' d='M19.5 8.25l-7.5 7.5-7.5-7.5'/%3e%3c/svg%3e")` }}
          >
            <option value="all">All Projects</option>
            {availableProjects.map((slug) => (
              <option key={slug} value={slug}>{humanizeProject(slug)}</option>
            ))}
          </select>
        </div>
      )}

      {/* Filter row */}
      <div className="flex items-center gap-1.5">
        <div className="flex items-center gap-1.5 overflow-x-auto flex-1 scrollbar-hide filter-scroll-fade">
          {primaryFilters.map((f) => (
            <button
              key={f.value}
              onClick={() => { onFilterChange(f.value); setShowMore(false); }}
              className={`whitespace-nowrap px-3.5 py-1.5 rounded-full text-[14px] font-medium transition-all duration-200 min-h-[36px] touch-manipulation ${
                filter === f.value
                  ? "bg-gold/[0.12] text-gold border border-gold/20 font-semibold"
                  : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated border border-transparent"
              }`}
            >
              {f.label}
            </button>
          ))}

          {/* More dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowMore(!showMore)}
              aria-haspopup="menu"
              aria-expanded={showMore}
              className={`whitespace-nowrap px-3.5 py-1.5 rounded-full text-[14px] font-medium transition-all duration-200 min-h-[36px] touch-manipulation flex items-center gap-1 ${
                isMoreActive
                  ? "bg-gold/[0.12] text-gold border border-gold/20 font-semibold"
                  : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated border border-transparent"
              }`}
            >
              {isMoreActive ? moreFilters.find((f) => f.value === filter)?.label : "More"}
              <svg className={`w-3 h-3 transition-transform ${showMore ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
              </svg>
            </button>
            {showMore && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setShowMore(false)} />
                <div role="menu" className="absolute top-full left-0 mt-1 z-20 bg-surface-elevated border border-edge rounded-lg shadow-lg py-1 min-w-[140px]">
                  {moreFilters.map((f) => (
                    <button
                      key={f.value}
                      role="menuitem"
                      onClick={() => { onFilterChange(f.value); setShowMore(false); }}
                      className={`w-full text-left px-3 py-2 text-[14px] transition-colors ${
                        filter === f.value
                          ? "text-gold font-medium"
                          : "text-ink-secondary hover:text-ink hover:bg-surface-hover"
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                  <div className="border-t border-edge my-1" />
                  <button
                    onClick={() => { setShowIngest(true); setShowMore(false); setIngestError(null); setIngestSuccess(null); }}
                    className="w-full text-left px-3 py-2 text-[14px] text-ink-secondary hover:text-ink hover:bg-surface-hover transition-colors flex items-center gap-1.5"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244" />
                    </svg>
                    Ingest URL
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Sort dropdown */}
        <div className="relative shrink-0">
          <button
            onClick={() => setShowSortMenu(!showSortMenu)}
            aria-haspopup="menu"
            aria-expanded={showSortMenu}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[14px] text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated transition-all touch-manipulation"
          >
            <span className="text-ink-faint text-[14px] font-mono uppercase tracking-wider">Sort:</span>
            <span className="font-medium">{sortLabels[sortBy]}</span>
            <svg className={`w-3 h-3 transition-transform ${showSortMenu ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>
          {showSortMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowSortMenu(false)} />
              <div role="menu" className="absolute top-full right-0 mt-1 z-20 bg-surface-elevated border border-edge rounded-lg shadow-lg py-1 min-w-[120px]">
                {(["priority", "newest", "oldest", "unread"] as SortOption[]).map((s) => (
                  <button
                    key={s}
                    role="menuitem"
                    onClick={() => { onSortChange(s); setShowSortMenu(false); }}
                    className={`w-full text-left px-3 py-2 text-[14px] transition-colors ${
                      sortBy === s
                        ? "text-gold font-medium"
                        : "text-ink-secondary hover:text-ink hover:bg-surface-hover"
                    }`}
                  >
                    {sortLabels[s]}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        {/* View mode toggle */}
        <div className="flex gap-0.5 p-0.5 bg-surface-elevated rounded-md shrink-0">
          <button
            onClick={() => onViewModeChange("timeline")}
            className={`p-1.5 rounded transition-colors ${
              viewMode === "timeline" ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-tertiary"
            }`}
            aria-label="Timeline view"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 16 16" stroke="currentColor" strokeWidth={1.5}>
              <path d="M2 3h12M2 6.5h12M2 10h12M2 13.5h12" />
            </svg>
          </button>
          <button
            onClick={() => onViewModeChange("summary")}
            className={`p-1.5 rounded transition-colors ${
              viewMode === "summary" ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-tertiary"
            }`}
            aria-label="Summary view"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 16 16" stroke="currentColor" strokeWidth={1.5}>
              <rect x="2" y="2" width="12" height="5" rx="1" />
              <path d="M2 10h5M2 13h8" />
            </svg>
          </button>
        </div>

        {/* Refresh */}
        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="p-1.5 rounded-lg text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated transition-all touch-manipulation shrink-0"
          aria-label="Refresh feed"
        >
          <svg
            className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182"
            />
          </svg>
        </button>

        {/* Show archived */}
        <button
          onClick={() => onShowArchivedChange(!showArchived)}
          className={`p-1.5 rounded-lg transition-all touch-manipulation shrink-0 ${
            showArchived
              ? "text-gold bg-gold/[0.08]"
              : "text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated"
          }`}
          aria-label={showArchived ? "Hide archived" : "Show archived"}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z" />
          </svg>
        </button>

        {/* Density */}
        <button
          onClick={() => onDensityChange(density === "default" ? "compact" : "default")}
          className="p-1.5 rounded-lg text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated transition-all touch-manipulation shrink-0"
          aria-label={`Switch to ${density === "default" ? "compact" : "default"} view`}
        >
          {density === "default" ? (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 16 16" stroke="currentColor" strokeWidth={1.5}>
              <path d="M2 3h12M2 6.5h12M2 10h12M2 13.5h12" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 16 16" stroke="currentColor" strokeWidth={1.5}>
              <rect x="2" y="2" width="12" height="4" rx="1" />
              <rect x="2" y="8" width="12" height="4" rx="1" />
            </svg>
          )}
        </button>
      </div>

      {/* Ingest URL bar - only visible when activated from More dropdown or keyboard shortcut */}
      {showIngest && (
        <div>
          <form
            className="flex items-center gap-1.5"
            onSubmit={async (e) => {
              e.preventDefault();
              const trimmed = ingestUrl.trim();
              if (!trimmed) return;
              setIngesting(true);
              setIngestError(null);
              setIngestSuccess(null);
              const result = await onIngestUrl(trimmed);
              if (result.success) {
                setIngestSuccess(`Ingested from ${result.domain}`);
                setIngestUrl("");
                setTimeout(() => { setShowIngest(false); setIngestSuccess(null); }, 3000);
              } else {
                setIngestError(result.error || "Ingestion failed");
              }
              setIngesting(false);
            }}
          >
            <input
              type="url"
              value={ingestUrl}
              onChange={(e) => setIngestUrl(e.target.value)}
              placeholder="https://..."
              disabled={ingesting}
              autoFocus
              className="flex-1 text-[14px] px-2.5 py-1.5 rounded-lg bg-surface-elevated border border-edge text-ink placeholder:text-ink-faint focus:border-gold/30 focus:outline-none transition-colors min-w-0"
            />
            <button
              type="submit"
              disabled={ingesting || !ingestUrl.trim()}
              className="text-[14px] font-medium px-3 py-1.5 rounded-lg bg-gold/[0.08] border border-gold/20 text-gold hover:bg-gold/[0.14] transition-colors touch-manipulation disabled:opacity-50 whitespace-nowrap"
            >
              {ingesting ? "Reading..." : "Ingest"}
            </button>
            <button
              type="button"
              onClick={() => { setShowIngest(false); setIngestError(null); setIngestSuccess(null); }}
              className="p-1.5 rounded-lg text-ink-faint hover:text-ink-tertiary hover:bg-surface-elevated transition-all touch-manipulation"
              aria-label="Close"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </form>
          {ingestError && (
            <p className="text-[14px] text-type-error mt-1">{ingestError}</p>
          )}
          {ingestSuccess && (
            <p className="text-[14px] text-type-task mt-1">{ingestSuccess}</p>
          )}
        </div>
      )}
    </div>
  );
}

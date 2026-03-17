import { useState } from "react";
import type { ResearchFinding, ResearchPlan, ResearchStats, CurationStatus } from "./types";
import {
  FILTERS,
  type Filter,
  FILTER_LABELS,
  PLAN_STATUS_STYLES,
  timeAgo,
  priorityBarColor,
  priorityPercent,
} from "./helpers";
import SubsystemPill from "./SubsystemPill";

export default function FindingsList({
  findings,
  plans,
  stats,
  total,
  selectedId,
  onSelect,
  onCurate,
  loading,
  hasMore,
  onLoadMore,
}: {
  findings: ResearchFinding[];
  plans: ResearchPlan[];
  stats: ResearchStats | null;
  total: number;
  selectedId: string | null;
  onSelect: (id: string, type: "finding" | "plan") => void;
  onCurate: (id: string, status: CurationStatus) => void;
  loading: boolean;
  hasMore: boolean;
  onLoadMore: () => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  const showPlans = filter === "all" || filter === "plans";
  const showFindings = filter !== "plans";

  const filteredFindings = showFindings
    ? findings.filter((f) => {
        if (filter === "starred") return f.curation_status === "starred";
        if (filter === "dismissed") return f.curation_status === "dismissed";
        if (filter === "high-priority" && f.priority_score < 12) return false;
        if (filter === "scanner" && f.source !== "sota_scanner") return false;
        if (filter === "manual" && f.source === "sota_scanner") return false;
        // Default "all" hides dismissed
        if (filter === "all" && f.curation_status === "dismissed") return false;
        return true;
      })
    : [];

  // Starred findings float to top
  const sortedFindings = [...filteredFindings].sort((a, b) => {
    if (a.curation_status === "starred" && b.curation_status !== "starred")
      return -1;
    if (b.curation_status === "starred" && a.curation_status !== "starred")
      return 1;
    return 0;
  });

  const filteredPlans = showPlans ? plans : [];

  const starredCount = findings.filter(
    (f) => f.curation_status === "starred"
  ).length;
  const dismissedCount = findings.filter(
    (f) => f.curation_status === "dismissed"
  ).length;

  function getFilterCount(f: Filter): number | undefined {
    if (f === "starred") return starredCount;
    if (f === "dismissed") return dismissedCount;
    if (f === "plans") return plans.length;
    if (f === "high-priority") return stats?.high_priority ?? 0;
    return undefined;
  }

  return (
    <div className="flex flex-col h-full">
      {/* Filter bar */}
      <div className="flex items-center gap-1.5 mb-3 pb-3 border-b border-edge flex-wrap">
        {FILTERS.map((f) => {
          const count = getFilterCount(f);
          // Hide dismissed filter if count is 0
          if (f === "dismissed" && dismissedCount === 0) return null;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-[14px] font-medium border transition-colors cursor-pointer ${
                filter === f
                  ? "bg-gold/10 text-gold border-gold/25"
                  : "bg-surface border-edge text-ink-tertiary hover:text-ink-secondary hover:bg-surface-hover"
              }`}
            >
              {FILTER_LABELS[f]}
              {count !== undefined && count > 0 ? ` (${count})` : ""}
            </button>
          );
        })}
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto space-y-1 min-h-0">
        {loading && findings.length === 0 && plans.length === 0 ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 rounded-lg skeleton" />
            ))}
          </div>
        ) : sortedFindings.length === 0 && filteredPlans.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <p className="text-[15px] text-ink-faint">
              No items match this filter.
            </p>
          </div>
        ) : (
          <>
            {/* Plans */}
            {filteredPlans.map((p) => (
              <button
                key={p.id}
                onClick={() => onSelect(p.id, "plan")}
                className={`w-full text-left px-4 py-3 rounded-lg border transition-colors cursor-pointer ${
                  selectedId === p.id
                    ? "border-gold/30 bg-gold/[0.05]"
                    : "border-transparent hover:bg-surface-hover"
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <svg
                    className="w-5 h-5 text-gold shrink-0"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={1.5}
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                    />
                  </svg>
                  <span className="text-[15px] font-medium text-ink truncate flex-1">
                    {p.title}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[12px] font-medium border shrink-0 ${PLAN_STATUS_STYLES[p.status]}`}
                  >
                    {p.status}
                  </span>
                </div>
              </button>
            ))}

            {/* Separator */}
            {filteredPlans.length > 0 && sortedFindings.length > 0 && (
              <div className="border-t border-edge my-2" />
            )}

            {/* Findings */}
            {sortedFindings.map((f) => {
              const title =
                f.technique ??
                f.content
                  .split("\n")[0]
                  ?.replace(/^#\s*/, "")
                  .slice(0, 80) ??
                "Untitled";
              const isStarred = f.curation_status === "starred";
              const isDismissed = f.curation_status === "dismissed";
              const isHighPri = f.priority_score >= 12;

              return (
                <div
                  key={f.id}
                  className={`group relative rounded-lg border transition-colors ${
                    selectedId === f.id
                      ? "border-gold/30 bg-gold/[0.05]"
                      : isHighPri && !isDismissed
                        ? "border-amber-500/15 hover:bg-surface-hover"
                        : isDismissed
                          ? "border-transparent opacity-60 hover:opacity-80 hover:bg-surface-hover"
                          : "border-transparent hover:bg-surface-hover"
                  }`}
                >
                  {/* Priority accent stripe for high-pri */}
                  {isHighPri && !isDismissed && (
                    <div
                      className={`absolute left-0 top-2 bottom-2 w-[3px] rounded-full ${priorityBarColor(f.priority_score)}`}
                    />
                  )}

                  <button
                    onClick={() => onSelect(f.id, "finding")}
                    className="w-full text-left px-4 py-3 cursor-pointer"
                  >
                    <div className="flex items-start gap-2.5">
                      {/* Star indicator */}
                      {isStarred && (
                        <svg
                          className="w-4 h-4 text-gold shrink-0 mt-0.5"
                          fill="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.006 5.404.434c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.434 2.082-5.005Z" />
                        </svg>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-[15px] font-medium text-ink leading-snug line-clamp-2">
                          {title}
                        </div>
                        <div className="flex items-center gap-2.5 mt-1.5 flex-wrap">
                          {f.subsystem && (
                            <SubsystemPill subsystem={f.subsystem} />
                          )}
                          {isHighPri && (
                            <span className="text-[13px] font-semibold text-type-reminder tabular-nums">
                              {f.priority_score}/25
                            </span>
                          )}
                          <span className="text-[13px] text-ink-faint ml-auto">
                            {timeAgo(f.created_at)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </button>

                  {/* Curation actions (visible on hover) */}
                  <div className="absolute right-2 top-2 hidden group-hover:flex items-center gap-1 bg-surface/90 backdrop-blur-sm rounded-lg border border-edge px-1 py-0.5">
                    {/* Star / Unstar */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onCurate(f.id, isStarred ? null : "starred");
                      }}
                      className="p-1.5 rounded-md hover:bg-surface-hover transition-colors cursor-pointer"
                      title={isStarred ? "Remove star" : "Star"}
                    >
                      <svg
                        className={`w-4 h-4 ${isStarred ? "text-gold" : "text-ink-faint"}`}
                        fill={isStarred ? "currentColor" : "none"}
                        viewBox="0 0 24 24"
                        strokeWidth={1.5}
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M11.48 3.499a.562.562 0 0 1 1.04 0l2.125 5.111a.563.563 0 0 0 .475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 0 0-.182.557l1.285 5.385a.562.562 0 0 1-.84.61l-4.725-2.885a.562.562 0 0 0-.586 0L6.982 20.54a.562.562 0 0 1-.84-.61l1.285-5.386a.562.562 0 0 0-.182-.557l-4.204-3.602a.562.562 0 0 1 .321-.988l5.518-.442a.563.563 0 0 0 .475-.345L11.48 3.5Z"
                        />
                      </svg>
                    </button>

                    {/* Dismiss / Restore */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onCurate(f.id, isDismissed ? null : "dismissed");
                      }}
                      className="p-1.5 rounded-md hover:bg-surface-hover transition-colors cursor-pointer"
                      title={isDismissed ? "Restore" : "Dismiss"}
                    >
                      {isDismissed ? (
                        <svg
                          className="w-4 h-4 text-ink-faint"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3"
                          />
                        </svg>
                      ) : (
                        <svg
                          className="w-4 h-4 text-ink-faint"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M6 18 18 6M6 6l12 12"
                          />
                        </svg>
                      )}
                    </button>

                    {/* Archive */}
                    {!isDismissed && f.curation_status !== "archived" && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onCurate(f.id, "archived");
                        }}
                        className="p-1.5 rounded-md hover:bg-surface-hover transition-colors cursor-pointer"
                        title="Archive (mark as reviewed)"
                      >
                        <svg
                          className="w-4 h-4 text-ink-faint"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>

      {/* Load more */}
      {hasMore && filter !== "plans" && (
        <div className="flex justify-center pt-3 border-t border-edge mt-2">
          <button
            onClick={onLoadMore}
            disabled={loading}
            className="px-4 py-2 rounded-lg text-[14px] font-medium text-ink-secondary hover:bg-surface-hover transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading
              ? "Loading..."
              : `Load more (${total - findings.length} remaining)`}
          </button>
        </div>
      )}
    </div>
  );
}

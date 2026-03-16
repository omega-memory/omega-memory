import type { ResearchFinding, CurationStatus } from "./types";
import { SOURCE_ICONS, timeAgo } from "./helpers";
import ScoreBadge from "./ScoreBadge";
import SubsystemPill from "./SubsystemPill";

export default function FindingDetail({
  finding,
  onCreatePlan,
  onCurate,
}: {
  finding: ResearchFinding | null;
  onCreatePlan: (finding: ResearchFinding) => void;
  onCurate: (id: string, status: CurationStatus) => void;
}) {
  if (!finding) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-16 text-center">
        <svg
          className="w-12 h-12 text-ink-faint/30 mb-4"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
          />
        </svg>
        <p className="text-[16px] text-ink-faint">
          Select a finding to view details
        </p>
      </div>
    );
  }

  const lines = finding.content.split("\n").filter(Boolean);
  const title =
    finding.technique ??
    lines[0]?.replace(/^#\s*/, "").slice(0, 120) ??
    "Untitled";
  const body = lines.slice(1).join("\n");

  const isStarred = finding.curation_status === "starred";
  const isDismissed = finding.curation_status === "dismissed";
  const isArchived = finding.curation_status === "archived";

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Title and metadata */}
      <div className="mb-5">
        <div className="flex items-start gap-3">
          <h2 className="text-[24px] font-semibold text-ink leading-snug flex-1">
            {title}
          </h2>

          {/* Curation status badge */}
          {isStarred && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[13px] font-medium bg-gold/10 text-gold border border-gold/25 shrink-0">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.006 5.404.434c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.434 2.082-5.005Z" />
              </svg>
              Starred
            </span>
          )}
          {isDismissed && (
            <span className="px-2.5 py-1 rounded-md text-[13px] font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/20 shrink-0">
              Dismissed
            </span>
          )}
          {isArchived && (
            <span className="px-2.5 py-1 rounded-md text-[13px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 shrink-0">
              Reviewed
            </span>
          )}
        </div>

        <div className="flex items-center gap-2.5 mt-3 flex-wrap">
          {finding.subsystem && (
            <SubsystemPill subsystem={finding.subsystem} />
          )}
          {finding.source === "sota_scanner" && (
            <span className="px-2.5 py-1 rounded-md text-[13px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Scanner
            </span>
          )}
          <span className="text-[14px] text-ink-faint">
            {SOURCE_ICONS[finding.source_type] ?? finding.source_type}
          </span>
          {finding.domain && (
            <span className="text-[14px] text-ink-faint">{finding.domain}</span>
          )}
          <span className="text-[14px] text-ink-faint">
            {timeAgo(finding.created_at)}
          </span>
        </div>
      </div>

      {/* Scores */}
      <div className="mb-5 p-4 rounded-lg bg-surface-elevated border border-edge">
        <ScoreBadge
          impact={finding.impact_score}
          feasibility={finding.feasibility_score}
          priority={finding.priority_score}
        />
      </div>

      {/* Full content */}
      <pre className="text-[16px] text-ink-secondary whitespace-pre-wrap font-sans leading-relaxed mb-5 flex-1">
        {body || finding.content}
      </pre>

      {/* Source URL */}
      {finding.source_url && (
        <a
          href={finding.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-[15px] text-gold hover:underline mb-5 cursor-pointer"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25"
            />
          </svg>
          View source
        </a>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 pt-4 border-t border-edge flex-wrap">
        <button
          onClick={() => onCreatePlan(finding)}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/25 hover:bg-gold/20 transition-colors cursor-pointer"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m3.75 9v6m3-3H9m1.5-12H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
          Create Plan
        </button>

        {/* Star / Unstar */}
        <button
          onClick={() =>
            onCurate(finding.id, isStarred ? null : "starred")
          }
          className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium border transition-colors cursor-pointer ${
            isStarred
              ? "bg-gold/10 text-gold border-gold/25 hover:bg-gold/20"
              : "text-ink-tertiary border-edge hover:bg-surface-hover"
          }`}
        >
          <svg
            className="w-4 h-4"
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
          {isStarred ? "Unstar" : "Star"}
        </button>

        {/* Dismiss / Restore */}
        <button
          onClick={() =>
            onCurate(finding.id, isDismissed ? null : "dismissed")
          }
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium text-ink-tertiary border border-edge hover:bg-surface-hover transition-colors cursor-pointer"
        >
          {isDismissed ? (
            <>
              <svg
                className="w-4 h-4"
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
              Restore
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4"
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
              Dismiss
            </>
          )}
        </button>

        {/* Archive */}
        {!isDismissed && (
          <button
            onClick={() =>
              onCurate(
                finding.id,
                isArchived ? null : "archived"
              )
            }
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium border transition-colors cursor-pointer ${
              isArchived
                ? "bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20"
                : "text-ink-tertiary border-edge hover:bg-surface-hover"
            }`}
          >
            <svg
              className="w-4 h-4"
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
            {isArchived ? "Reviewed" : "Mark Reviewed"}
          </button>
        )}
      </div>
    </div>
  );
}

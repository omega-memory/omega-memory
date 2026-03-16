import React from "react";
import type { InsightsData, ContentStats } from "../lib/types";
import { timeUntil } from "../contentUtils";
import PlatformIcon from "../shared/PlatformIcon";
import CollapsibleSection from "./CollapsibleSection";
import { formatContentType } from "./helpers";

function PipelineColumn({
  label,
  icon,
  stats,
}: {
  label: string;
  icon: "x" | "linkedin";
  stats: ContentStats;
}) {
  const generated = stats.pending + stats.approved + stats.published + stats.failed;
  const approved = stats.approved + stats.published;
  const published = stats.published;
  const approvedPct = generated > 0 ? Math.round((approved / generated) * 100) : 0;
  const publishedPct = generated > 0 ? Math.round((published / generated) * 100) : 0;

  const topTypes = Object.entries(stats.byContentType || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  const maxTypeCount = topTypes[0]?.[1] || 1;

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-3">
        <PlatformIcon platform={icon} size={16} />
        <span className="text-[16px] font-medium text-ink-secondary">{label}</span>
      </div>

      {generated === 0 ? (
        <p className="text-[16px] text-ink-tertiary">No content</p>
      ) : (
        <div className="space-y-3">
          <div className="space-y-2">
            <div className="text-[16px] text-ink tabular-nums font-medium">{generated} generated</div>

            <div>
              <div className="flex justify-between mb-1 text-[16px]">
                <span className="text-ink-secondary">Approved</span>
                <span className="tabular-nums">
                  <span className="text-type-decision font-medium">{approvedPct}%</span>
                  <span className="text-ink-tertiary ml-1">({approved})</span>
                </span>
              </div>
              <div className="h-2 bg-surface-elevated rounded-full overflow-hidden">
                <div className="h-full rounded-full bg-type-decision/40" style={{ width: `${approvedPct}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1 text-[16px]">
                <span className="text-ink-secondary">Published</span>
                <span className="tabular-nums">
                  <span className="text-type-lesson font-medium">{publishedPct}%</span>
                  <span className="text-ink-tertiary ml-1">({published})</span>
                </span>
              </div>
              <div className="h-2 bg-surface-elevated rounded-full overflow-hidden">
                <div className="h-full rounded-full bg-type-lesson" style={{ width: `${publishedPct}%` }} />
              </div>
            </div>
          </div>

          {(stats.failed > 0 || stats.retries > 0) && (
            <div className="flex items-center gap-3 text-[16px]">
              {stats.failed > 0 && (
                <span className="text-type-error tabular-nums">{stats.failed} failed</span>
              )}
              {stats.retries > 0 && (
                <span className="text-ink-tertiary tabular-nums">{stats.retries} retries</span>
              )}
            </div>
          )}

          <div className="text-[16px]">
            {stats.queueDepth > 0 ? (
              <div className="space-y-0.5">
                <span className="text-ink-secondary">
                  Scheduled: <span className="tabular-nums font-medium">{stats.queueDepth}</span> post{stats.queueDepth !== 1 ? "s" : ""}
                </span>
                {stats.nextScheduled && (
                  <div className="text-ink-tertiary">
                    Next: {timeUntil(stats.nextScheduled)}
                  </div>
                )}
              </div>
            ) : (
              <span className="text-ink-tertiary">No posts scheduled</span>
            )}
          </div>

          {topTypes.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[14px] text-ink-tertiary uppercase tracking-wider">Top types</span>
              {topTypes.map(([type, count]) => (
                <div key={type} className="flex items-center gap-2">
                  <span className="text-[16px] text-ink-secondary w-20 truncate">
                    {formatContentType(type)}
                  </span>
                  <div className="flex-1 h-3 bg-surface-elevated rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-type-decision/40 transition-all duration-500"
                      style={{ width: `${(count / maxTypeCount) * 100}%` }}
                    />
                  </div>
                  <span className="text-[14px] text-ink-tertiary tabular-nums w-5 text-right">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Publishing({ content }: { content: NonNullable<InsightsData["content"]> }) {
  const hasAny =
    content.tweets.pending + content.tweets.approved + content.tweets.published + content.tweets.failed + content.tweets.rejected > 0 ||
    content.linkedin.pending + content.linkedin.approved + content.linkedin.published + content.linkedin.failed + content.linkedin.rejected > 0;

  const totalPublished = content.tweets.published + content.linkedin.published;
  const totalPending = content.tweets.pending + content.linkedin.pending;
  const summaryText = hasAny
    ? `${totalPublished} published, ${totalPending} pending`
    : "No content yet";

  return (
    <CollapsibleSection label="Publishing" summary={summaryText}>
      {!hasAny ? (
        <p className="text-[16px] text-ink-tertiary">No content yet</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <PipelineColumn label="X" icon="x" stats={content.tweets} />
          <PipelineColumn label="LinkedIn" icon="linkedin" stats={content.linkedin} />
        </div>
      )}
    </CollapsibleSection>
  );
}

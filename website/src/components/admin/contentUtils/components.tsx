import React from "react";

import {
  ParsedContent,
  ContentCategory,
  GRANT_STATUS_STYLES,
  LeaderboardEntry,
} from "./types";

import {
  humanizeBullet,
  summarizeDetail,
  generateExecutiveSummary,
  detectKeyValueBullets,
  splitNumberedContent,
  detectSections,
  parseGrantData,
  parseLeaderboard,
} from "./parse";

import { humanizeSectionHeading } from "./format";

// ─── React Components ───────────────────────────────────────

export function renderInline(text: string): (string | React.ReactElement)[] {
  const parts: (string | React.ReactElement)[] = [];
  const regex = /(\*\*(.+?)\*\*)|(`([^`]+)`)|(https?:\/\/[^\s<>)]+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(
        <strong key={`b${match.index}`} className="text-ink font-medium">
          {match[2]}
        </strong>
      );
    } else if (match[4]) {
      parts.push(
        <code
          key={`c${match.index}`}
          className="text-[13px] bg-surface-elevated text-ink-secondary px-1 py-0.5 rounded border border-edge-subtle"
        >
          {match[4]}
        </code>
      );
    } else if (match[5]) {
      const displayUrl = match[5].replace(/^https?:\/\//, '').replace(/\/$/, '');
      parts.push(
        <a key={`u${match.index}`} href={match[5]}
           target="_blank" rel="noopener noreferrer"
           className="text-type-decision hover:text-type-decision/80 underline decoration-type-decision/30 transition-colors">
          {displayUrl.length > 50 ? displayUrl.slice(0, 47) + '...' : displayUrl}
        </a>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

export function SimpleMd({ text, humanize = false, maxBullets = 0 }: { text: string; humanize?: boolean; maxBullets?: number }) {
  const lines = text.split("\n");
  const elements: React.ReactElement[] = [];
  let i = 0;
  let bulletCount = 0;
  let bulletOverflow = 0;

  const h = (line: string) => humanize ? humanizeBullet(line) : line;

  while (i < lines.length) {
    const line = lines[i];

    // Code block
    if (line.trimStart().startsWith("```")) {
      i++;
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) i++;
      i++;
      if (!humanize) {
        // skipped
      }
      continue;
    }

    // Skip technical lines in humanize mode
    if (humanize) {
      const trimmed = line.trim();
      if (/^\/?\w+\/[\w/]+\.\w+\s*:/.test(trimmed)) { i++; continue; }
      if (/^src\//.test(trimmed)) { i++; continue; }
      if (/^\s*(cd |python3? |npm |git |bash |sh |curl |pip )/.test(trimmed)) { i++; continue; }
      if ((trimmed.match(/--\w+/g) || []).length >= 2) { i++; continue; }
    }

    // Headers
    if (/^###\s+/.test(line)) {
      elements.push(
        <h4
          key={elements.length}
          className="text-[16px] font-semibold text-ink mt-3 mb-1"
        >
          {renderInline(h(line.replace(/^###\s+/, "")))}
        </h4>
      );
      i++;
      continue;
    }
    if (/^##\s+/.test(line)) {
      elements.push(
        <h4
          key={elements.length}
          className="text-[16px] font-semibold text-ink mt-3 mb-1"
        >
          {renderInline(h(line.replace(/^##\s+/, "")))}
        </h4>
      );
      i++;
      continue;
    }

    // Bullet points
    if (/^\s*[-*]\s+/.test(line)) {
      if (humanize && maxBullets > 0) {
        bulletCount++;
        if (bulletCount > maxBullets) {
          bulletOverflow++;
          i++;
          continue;
        }
      }
      const bulletText = h(line.replace(/^\s*[-*]\s+/, ""));
      if (bulletText.trim()) {
        elements.push(
          <div key={elements.length} className="flex gap-2 mt-0.5">
            <span className="text-ink-faint shrink-0 mt-[1px]">&ndash;</span>
            <span className="text-[15px] text-ink-secondary leading-relaxed">
              {renderInline(bulletText)}
            </span>
          </div>
        );
      }
      i++;
      continue;
    }

    // Empty line
    if (line.trim() === "") {
      elements.push(<div key={elements.length} className="h-2" />);
      i++;
      continue;
    }

    // Regular paragraph
    const pText = h(line);
    if (pText.trim()) {
      elements.push(
        <p
          key={elements.length}
          className="text-[15px] text-ink-secondary leading-relaxed"
        >
          {renderInline(pText)}
        </p>
      );
    }
    i++;
  }

  if (bulletOverflow > 0) {
    elements.push(
      <span key="overflow" className="text-[11px] text-ink-faint ml-5">
        +{bulletOverflow} more
      </span>
    );
  }

  return <div>{elements}</div>;
}

export function SummaryBullets({
  text,
  parsed,
}: {
  text: string;
  parsed: ParsedContent;
}) {
  const bullets = summarizeDetail(text, parsed);
  if (bullets.length === 0) return null;
  const shown = bullets.slice(0, 3);
  const remaining = bullets.length - shown.length;
  return (
    <div className="space-y-1">
      {shown.map((bullet, i) => (
        <div key={i} className="flex gap-2">
          <span className="text-ink-faint shrink-0 mt-[2px] text-[12px]">
            &ndash;
          </span>
          <span className="text-[14px] text-ink-secondary leading-relaxed">
            {renderInline(humanizeBullet(bullet))}
          </span>
        </div>
      ))}
      {remaining > 0 && (
        <span className="text-[11px] text-ink-faint ml-5">
          +{remaining} more
        </span>
      )}
    </div>
  );
}

export function CategoryIcon({ category }: { category: ContentCategory }) {
  const cls = "w-3 h-3 inline-block shrink-0";
  switch (category) {
    case "checkpoint":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M3 1v14l1-.5 3-2 3 2 3-2 1 .5V1H3zm2 2h6v2H5V3z" />
        </svg>
      );
    case "benchmark":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M2 14V8h3v6H2zm4.5 0V4h3v10h-3zM11 14V1h3v13h-3z" />
        </svg>
      );
    case "entity":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M2 14V3h5v2h5v9H2zm2-9v2h2V5H4zm4 2h2V5H8v2zm-4 4h2V9H4v2zm4 0h2V9H8v2z" />
        </svg>
      );
    case "grant":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M4 1h8l1 1v12l-1 1H4l-1-1V2l1-1zm1 2v2h6V3H5zm0 4v1h6V7H5zm0 3v1h4v-1H5z" />
        </svg>
      );
    case "financial":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.5 3v1.2c1.1.2 2 .8 2 1.8 0 1.2-1 1.8-2.5 2-.7.1-1 .4-1 .7 0 .3.3.6 1 .6.6 0 1.1-.2 1.5-.5l.8 1c-.5.5-1.2.8-1.8.9V13h-1v-1.2c-1.2-.2-2-.9-2-1.9s1-1.8 2.5-2c.7-.1 1-.4 1-.7s-.3-.5-.9-.5c-.5 0-1 .2-1.4.5l-.8-1c.5-.4 1.1-.7 1.6-.8V4h1z" />
        </svg>
      );
    case "deadline":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2a5 5 0 110 10A5 5 0 018 3zm-.5 2v3.5l2.5 1.5.5-.9-2-1.2V5h-1z" />
        </svg>
      );
    case "stats":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M1 14h14v1H1v-1zM3 8h2v5H3V8zm4-4h2v9H7V4zm4 2h2v7h-2V6z" />
        </svg>
      );
    default:
      return null;
  }
}

export function DocTypeIcon({ kind }: { kind: string }) {
  const cls = "w-4 h-4";
  switch (kind) {
    case "benchmark":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M2 14V8h3v6H2zm4.5 0V4h3v10h-3zM11 14V1h3v13h-3z" />
        </svg>
      );
    case "stats":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M1 14h14v1H1v-1zM3 8h2v5H3V8zm4-4h2v9H7V4zm4 2h2v7h-2V6z" />
        </svg>
      );
    case "file":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M4 1h5.5L13 4.5V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1Zm5 1v3h3L9 2ZM5 7h6v1H5V7Zm0 2.5h6v1H5v-1Zm0 2.5h4v1H5v-1Z" />
        </svg>
      );
    default:
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M4 1h8a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1Zm1 3v1h6V4H5Zm0 3v1h6V7H5Zm0 3v1h4v-1H5Z" />
        </svg>
      );
  }
}

export function KeyValueList({ items }: { items: { key: string; value: string }[] }) {
  return (
    <div className="mt-4 expand-enter grid gap-2.5">
      {items.map((item, i) => (
        <div key={i} className="rounded-lg bg-surface-elevated border border-edge-subtle p-3.5 flex flex-col gap-1">
          <span className="text-[12px] font-semibold text-ink-tertiary uppercase tracking-wider">{item.key}</span>
          <span className="text-[15px] text-ink leading-relaxed break-words">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

const NUMBERED_ACCENT_COLORS = [
  "from-type-decision/80 to-type-decision/40",
  "from-type-lesson/80 to-type-lesson/40",
  "from-type-reminder/80 to-type-reminder/40",
  "from-type-task/80 to-type-task/40",
  "from-type-preference/80 to-type-preference/40",
  "from-type-decision/60 to-type-decision/30",
];

const NUMBERED_BG_COLORS = [
  "bg-type-decision/8",
  "bg-type-lesson/8",
  "bg-type-reminder/8",
  "bg-type-task/8",
  "bg-type-preference/8",
  "bg-type-decision/6",
];

const NUMBERED_NUM_COLORS = [
  "text-type-decision",
  "text-type-lesson",
  "text-type-reminder",
  "text-type-task",
  "text-type-preference",
  "text-type-decision/80",
];

export function NumberedList({ items }: { items: string[] }) {
  return (
    <div className="mt-4 expand-enter space-y-2">
      {items.map((item, i) => {
        const colorIdx = i % NUMBERED_ACCENT_COLORS.length;
        return (
          <div key={i} className={`relative rounded-lg ${NUMBERED_BG_COLORS[colorIdx]} border border-edge-subtle overflow-hidden`}>
            <div className={`absolute left-0 top-0 bottom-0 w-[3px] bg-gradient-to-b ${NUMBERED_ACCENT_COLORS[colorIdx]}`} />
            <div className="flex items-start gap-3.5 pl-4 pr-4 py-3">
              <span className={`text-[18px] font-bold tabular-nums shrink-0 leading-none mt-0.5 ${NUMBERED_NUM_COLORS[colorIdx]}`}>
                {i + 1}
              </span>
              <span className="text-[15px] text-ink leading-relaxed">{renderInline(item)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function SectionContent({ lines }: { lines: string[]; parsed?: ParsedContent }) {
  const text = lines.join('\n');

  const kvItems = detectKeyValueBullets(text);
  if (kvItems.length >= 2) {
    return <KeyValueList items={kvItems} />;
  }

  const numbered = splitNumberedContent(text);
  if (numbered.length >= 2) {
    return <NumberedList items={numbered} />;
  }

  const trimmed = text.trim();
  if (!trimmed) return null;
  return <SimpleMd text={trimmed} humanize maxBullets={12} />;
}

export function GrantStatsBar({ content }: { content: string }) {
  const grant = parseGrantData(content);
  const hasData = grant.status || grant.amount || grant.score;
  if (!hasData) return null;

  return (
    <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
      {grant.status && (() => {
        const s = GRANT_STATUS_STYLES[grant.status] || GRANT_STATUS_STYLES.draft;
        return (
          <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full ${s.bg} ${s.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
            {s.label}
          </span>
        );
      })()}
      {grant.amount && (
        <span className="inline-flex items-center text-[12px] font-bold tabular-nums px-2 py-0.5 rounded-full bg-gold/[0.06] text-gold border border-gold/10">
          {grant.amount}
        </span>
      )}
      {grant.score && (
        <span className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-type-decision/[0.08] text-type-decision">
          <span className="tabular-nums">{grant.score.value}/{grant.score.max}</span>
          <span className="flex gap-px ml-0.5">
            {Array.from({ length: grant.score.max }, (_, i) => (
              <span
                key={i}
                className={`w-[3px] h-[10px] rounded-sm ${
                  i < grant.score!.value
                    ? "bg-type-decision"
                    : "bg-type-decision/20"
                }`}
              />
            ))}
          </span>
        </span>
      )}
      {grant.funder && (
        <span className="text-[11px] text-ink-tertiary font-medium px-1.5 py-0.5 rounded-full bg-surface-elevated border border-edge-subtle">
          {grant.funder}
        </span>
      )}
    </div>
  );
}

function LeaderboardCard({ entries }: { entries: LeaderboardEntry[] }) {
  return (
    <div className="rounded-lg border border-edge-subtle overflow-hidden">
      <div className="px-3 py-2 bg-surface border-b border-edge-subtle">
        <span className="text-[11px] font-semibold text-ink-tertiary uppercase tracking-wider">Rankings</span>
      </div>
      <div className="divide-y divide-edge-subtle">
        {entries.map((entry) => {
          const barWidth = entry.score !== null ? Math.max((entry.score / 100) * 100, 8) : 0;
          const isTop = entry.rank === 1;
          return (
            <div
              key={entry.name}
              className={`relative flex items-center gap-3 px-3 py-2.5 transition-colors ${
                entry.isOmega ? "bg-gold/[0.04]" : ""
              }`}
            >
              {/* Rank */}
              <span className={`w-5 text-center text-[13px] font-bold tabular-nums shrink-0 ${
                isTop ? "text-gold" : entry.isOmega ? "text-type-decision" : "text-ink-faint"
              }`}>
                {entry.rank}
              </span>

              {/* Name + bar */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-[13px] font-medium truncate ${
                    entry.isOmega ? "text-ink font-semibold" : "text-ink-secondary"
                  }`}>
                    {entry.name}
                    {entry.isOmega && (
                      <span className="ml-1.5 text-[10px] font-semibold text-gold bg-gold/10 px-1.5 py-0.5 rounded-full align-middle">
                        YOU
                      </span>
                    )}
                  </span>
                  <span className={`text-[13px] tabular-nums shrink-0 ml-2 ${
                    entry.isOmega ? "font-bold text-ink" : "font-medium text-ink-tertiary"
                  }`}>
                    {entry.score !== null ? `${entry.score}%` : "-"}
                  </span>
                </div>
                {entry.score !== null && (
                  <div className="h-1 bg-edge-subtle rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all bar-animate ${
                        entry.isOmega
                          ? "bg-gradient-to-r from-gold/80 to-gold/40"
                          : isTop
                            ? "bg-gradient-to-r from-type-lesson/60 to-type-lesson/30"
                            : "bg-gradient-to-r from-ink-faint/40 to-ink-faint/20"
                      }`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RadialGauge({ value, label, unit }: { value: number; label: string; unit?: string }) {
  const size = 120;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.min(Math.max(value, 0), 100);
  const offset = circumference - (pct / 100) * circumference;
  const color = pct >= 80 ? 'var(--color-type-lesson)' : pct >= 60 ? 'var(--color-type-reminder)' : 'var(--color-type-error)';

  return (
    <div className="gauge-container flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="gauge-arc"
        />
        <text x={size / 2} y={size / 2} textAnchor="middle" dominantBaseline="central">
          <tspan fill="var(--color-ink)" fontSize="28" fontWeight="700">{value}</tspan>
          {unit && <tspan fill="var(--color-ink-tertiary)" fontSize="14">{unit}</tspan>}
        </text>
      </svg>
      <span className="text-[12px] text-ink-tertiary mt-1 uppercase tracking-wider">{label}</span>
    </div>
  );
}

export function MetricsView({
  parsed,
  raw,
}: {
  parsed: ParsedContent;
  raw: string;
}) {
  const hasMetrics = parsed.metrics.length > 0;

  const headlineLabels = ["overall", "task-averaged", "total", "average"];
  const headline = parsed.metrics.find((m) =>
    headlineLabels.includes(m.label.toLowerCase())
  );
  const breakdown = parsed.metrics.filter((m) => m !== headline);
  const summary = generateExecutiveSummary(parsed, null, raw);
  const omegaScore = headline ? parseFloat(headline.value) : null;
  const leaderboard = parsed.category === "benchmark" ? parseLeaderboard(raw, omegaScore) : null;

  return (
    <div className="expand-enter space-y-4">
      {/* Executive Summary */}
      {summary && (
        <div className="p-3 rounded-lg bg-surface border border-edge-subtle">
          <p className="text-[14px] text-ink-secondary leading-relaxed">{summary}</p>
        </div>
      )}

      {/* Radial gauge */}
      {headline && headline.unit === "%" ? (
        <div className="flex justify-center">
          <RadialGauge
            value={parseFloat(headline.value)}
            label={headline.label}
            unit={headline.unit}
          />
        </div>
      ) : headline ? (
        <div className="bg-surface-elevated rounded-xl p-4 border border-edge-subtle text-center">
          <div className="text-[13px] text-ink-tertiary uppercase tracking-wider mb-1">
            {headline.label}
          </div>
          <div className="text-[32px] font-bold text-ink tabular-nums leading-none">
            {headline.value}
            {headline.unit && (
              <span className="text-[18px] text-ink-tertiary ml-0.5">
                {headline.unit}
              </span>
            )}
          </div>
        </div>
      ) : null}

      {/* Leaderboard */}
      {leaderboard && <LeaderboardCard entries={leaderboard} />}

      {/* Breakdown bars */}
      {breakdown.length > 0 && (
        <div className="space-y-2">
          {breakdown.map((m, i) => {
            const pct = m.unit === '%' ? Math.min(parseFloat(m.value), 100) : null;
            const barColor = pct !== null
              ? pct >= 80 ? 'from-type-lesson/80 to-type-lesson/40'
                : pct >= 60 ? 'from-type-reminder/80 to-type-reminder/40'
                : 'from-type-error/80 to-type-error/40'
              : 'from-type-lesson/80 to-type-lesson/40';

            return (
              <div key={i} className="bg-surface-elevated rounded-lg p-3 border border-edge-subtle">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[12px] text-ink-tertiary">{m.label}</span>
                  <span className="text-[15px] font-semibold text-ink tabular-nums">
                    {m.value}
                    {m.unit && <span className="text-[12px] text-ink-tertiary ml-0.5">{m.unit}</span>}
                  </span>
                </div>
                {pct !== null && (
                  <div className="h-1.5 bg-edge-subtle rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${barColor} bar-animate`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Deep Dive: full content */}
      {parsed.detail && (
        <div className="pt-4 border-t border-edge-subtle">
          <div className="text-[11px] font-semibold text-ink-tertiary uppercase tracking-wider mb-2">Details</div>
          <SimpleMd text={parsed.detail} humanize maxBullets={20} />
        </div>
      )}
      {!hasMetrics && !parsed.detail && <SummaryBullets text={raw} parsed={parsed} />}
    </div>
  );
}

export function CheckpointView({
  parsed,
  raw,
}: {
  parsed: ParsedContent;
  raw: string;
}) {
  if (parsed.sections.length === 0) {
    return <DefaultView parsed={parsed} raw={raw} />;
  }
  return (
    <div className="space-y-2 mt-3 expand-enter">
      {parsed.sections.map((s, i) => (
        <div
          key={i}
          className="bg-surface-elevated rounded-lg p-3 border border-edge-subtle"
        >
          <div className="text-[12px] font-semibold text-ink-secondary uppercase tracking-wider mb-1.5">
            {humanizeSectionHeading(s.heading)}
          </div>
          <SimpleMd text={s.body} humanize maxBullets={5} />
        </div>
      ))}
    </div>
  );
}

export function EntityView({
  parsed,
  raw,
}: {
  parsed: ParsedContent;
  raw: string;
}) {
  const hasKV = parsed.keyValues.length > 0;
  return (
    <div className="mt-3 expand-enter">
      {hasKV && (
        <div className="bg-surface-elevated rounded-lg border border-edge-subtle overflow-hidden mb-3">
          {parsed.keyValues.map((kv, i) => (
            <div
              key={i}
              className={`flex items-center justify-between px-3 py-2 ${
                i > 0 ? "border-t border-edge-subtle" : ""
              }`}
            >
              <span className="text-[13px] text-ink-tertiary">{kv.key}</span>
              <span className="text-[14px] text-ink font-medium tabular-nums">
                {kv.value}
              </span>
            </div>
          ))}
        </div>
      )}
      {parsed.detail ? (
        <SummaryBullets text={parsed.detail} parsed={parsed} />
      ) : (
        !hasKV && <SummaryBullets text={raw} parsed={parsed} />
      )}
    </div>
  );
}

export function GrantView({
  parsed,
  raw,
}: {
  parsed: ParsedContent;
  raw: string;
}) {
  const grant = parseGrantData(raw);
  const hasHero = grant.status || grant.amount || grant.score;

  return (
    <div className="mt-3 expand-enter">
      {/* Hero stat cards */}
      {hasHero && (
        <div className="grid grid-cols-3 gap-2 mb-3">
          {/* Status card */}
          {grant.status && (() => {
            const s = GRANT_STATUS_STYLES[grant.status] || GRANT_STATUS_STYLES.draft;
            return (
              <div className={`rounded-lg p-3 border border-edge-subtle ${s.bg} relative overflow-hidden`}>
                <div className={`absolute top-0 left-0 w-full h-[2px] ${s.dot}`} />
                <div className="text-[10px] text-ink-tertiary uppercase tracking-widest mb-1.5">Status</div>
                <div className={`text-[16px] font-bold ${s.text} flex items-center gap-1.5`}>
                  <span className={`w-2 h-2 rounded-full ${s.dot} shrink-0`} />
                  {s.label}
                </div>
              </div>
            );
          })()}

          {/* Amount card */}
          {grant.amount && (
            <div className="rounded-lg p-3 border border-gold/15 bg-gold/[0.04] relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gold/40" />
              <div className="text-[10px] text-ink-tertiary uppercase tracking-widest mb-1.5">Amount</div>
              <div className="text-[20px] font-bold text-gold tabular-nums leading-none">
                {grant.amount}
              </div>
            </div>
          )}

          {/* Score card */}
          {grant.score && (
            <div className="rounded-lg p-3 border border-type-decision/15 bg-type-decision/[0.04] relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[2px] bg-type-decision/40" />
              <div className="text-[10px] text-ink-tertiary uppercase tracking-widest mb-1.5">Score</div>
              <div className="flex items-end gap-1.5">
                <span className="text-[20px] font-bold text-type-decision tabular-nums leading-none">
                  {grant.score.value}
                </span>
                <span className="text-[13px] text-type-decision/50 font-medium mb-0.5">
                  /{grant.score.max}
                </span>
              </div>
              {/* Visual score bar */}
              <div className="flex gap-[3px] mt-2">
                {Array.from({ length: grant.score.max }, (_, i) => (
                  <div
                    key={i}
                    className={`flex-1 h-[4px] rounded-sm transition-all ${
                      i < grant.score!.value
                        ? "bg-type-decision"
                        : "bg-type-decision/15"
                    }`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Secondary info row: funder, duration, deadline */}
      {(grant.funder || grant.duration || grant.deadline) && (
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          {grant.funder && (
            <span className="inline-flex items-center gap-1.5 text-[12px] font-medium text-ink-secondary px-2.5 py-1 rounded-lg bg-surface-elevated border border-edge-subtle">
              <svg className="w-3.5 h-3.5 text-ink-tertiary" viewBox="0 0 16 16" fill="currentColor">
                <path d="M4 1h8l1 1v12l-1 1H4l-1-1V2l1-1zm1 2v2h6V3H5zm0 4v1h6V7H5zm0 3v1h4v-1H5z" />
              </svg>
              {grant.funder}
            </span>
          )}
          {grant.duration && (
            <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-tertiary px-2.5 py-1 rounded-lg bg-surface-elevated border border-edge-subtle">
              <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2a5 5 0 110 10A5 5 0 018 3zm-.5 2v3.5l2.5 1.5.5-.9-2-1.2V5h-1z" />
              </svg>
              {grant.duration}
            </span>
          )}
          {grant.deadline && (
            <span className="inline-flex items-center gap-1.5 text-[12px] text-type-reminder px-2.5 py-1 rounded-lg bg-type-reminder/[0.06] border border-type-reminder/10">
              <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path d="M4 0v2H2a1 1 0 00-1 1v11a1 1 0 001 1h12a1 1 0 001-1V3a1 1 0 00-1-1h-2V0h-2v2H6V0H4zM2 6h12v8H2V6z" />
              </svg>
              {grant.deadline}
            </span>
          )}
        </div>
      )}

      {/* Remaining parsed metrics not already shown */}
      {parsed.metrics.length > 0 && (
        <div className="grid grid-cols-2 gap-2 mb-3">
          {parsed.metrics.map((m, i) => (
            <div
              key={i}
              className="bg-surface-elevated rounded-lg p-2.5 border border-edge-subtle"
            >
              <div className="text-[10px] text-ink-tertiary uppercase tracking-widest mb-1">
                {m.label}
              </div>
              <div className="text-[17px] font-semibold text-ink tabular-nums">
                {m.value}
                {m.unit && (
                  <span className="text-[12px] text-ink-tertiary ml-0.5">
                    {m.unit}
                  </span>
                )}
              </div>
              {m.unit === "%" && (
                <div className="mt-1.5 h-1 bg-edge-subtle rounded-full overflow-hidden">
                  <div
                    className="h-full bg-type-lesson/60 rounded-full transition-all"
                    style={{ width: `${Math.min(parseFloat(m.value), 100)}%` }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Key-value pairs */}
      {parsed.keyValues.length > 0 && (
        <div className="bg-surface-elevated rounded-lg border border-edge-subtle overflow-hidden mb-3">
          {parsed.keyValues.map((kv, i) => (
            <div
              key={i}
              className={`flex items-center justify-between px-3 py-2 ${
                i > 0 ? "border-t border-edge-subtle" : ""
              }`}
            >
              <span className="text-[12px] text-ink-tertiary">{kv.key}</span>
              <span className="text-[14px] text-ink font-medium tabular-nums">
                {kv.value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Detail bullets */}
      {parsed.detail ? (
        <SummaryBullets text={parsed.detail} parsed={parsed} />
      ) : (
        <SummaryBullets text={raw} parsed={parsed} />
      )}
    </div>
  );
}

function BulletCard({ text, index }: { text: string; index: number }) {
  const humanized = humanizeBullet(text);
  // Detect special content types for visual callouts
  const hasEmail = /\b[\w.-]+@[\w.-]+\.\w+\b/.test(humanized);
  const hasAddress = /\b\d+[-\s]?\d*\s+\w+\s+(Dr|St|Ave|Rd|Blvd|Ln|Ct|Way|Pl)\b/i.test(humanized);
  const hasMoney = /[$\u20ac]\s?\d/.test(humanized);
  const hasDate = /\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d/i.test(humanized) || /\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b/.test(humanized);
  const hasDeadline = /\b(deadline|overdue|due|urgent|critical)\b/i.test(humanized);
  const hasWarning = /\bNOT\b/.test(text) || hasDeadline;

  const iconColor = hasWarning ? "text-type-error" : hasEmail ? "text-type-preference" : hasAddress ? "text-type-task" : hasMoney ? "text-type-reminder" : hasDate ? "text-type-decision" : "text-ink-faint";
  const borderColor = hasWarning ? "border-type-error/20" : "border-edge-subtle";
  const bgColor = hasWarning ? "bg-type-error/[0.04]" : "bg-surface-elevated";

  return (
    <div className={`rounded-lg ${bgColor} border ${borderColor} p-3.5 flex items-start gap-3`}>
      <span className={`shrink-0 mt-0.5 ${iconColor}`}>
        {hasWarning ? (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1L1 14h14L8 1zm0 4v4h0V5zm0 6a1 1 0 100 2 1 1 0 000-2z"/></svg>
        ) : hasEmail ? (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M2 3h12a1 1 0 011 1v8a1 1 0 01-1 1H2a1 1 0 01-1-1V4a1 1 0 011-1zm0 2v1l6 3 6-3V5L8 8 2 5z"/></svg>
        ) : hasAddress ? (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a5 5 0 015 5c0 3.5-5 9-5 9S3 9.5 3 6a5 5 0 015-5zm0 3a2 2 0 100 4 2 2 0 000-4z"/></svg>
        ) : hasMoney ? (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.5 3v1.2c1.1.2 2 .8 2 1.8 0 1.2-1 1.8-2.5 2-.7.1-1 .4-1 .7 0 .3.3.6 1 .6.6 0 1.1-.2 1.5-.5l.8 1c-.5.5-1.2.8-1.8.9V13h-1v-1.2c-1.2-.2-2-.9-2-1.9s1-1.8 2.5-2c.7-.1 1-.4 1-.7s-.3-.5-.9-.5c-.5 0-1 .2-1.4.5l-.8-1c.5-.4 1.1-.7 1.6-.8V4h1z"/></svg>
        ) : hasDate ? (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M4 0v2H2a1 1 0 00-1 1v11a1 1 0 001 1h12a1 1 0 001-1V3a1 1 0 00-1-1h-2V0h-2v2H6V0H4zM2 6h12v8H2V6zm2 2v2h2V8H4zm4 0v2h2V8H8zm4 0v2h2V8h-2z"/></svg>
        ) : (
          <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.5 3h1v5h-1V4zm0 7h1v1h-1v-1z"/></svg>
        )}
      </span>
      <span className="text-[15px] text-ink leading-relaxed">{renderInline(humanized)}</span>
    </div>
  );
}

export function DefaultView({
  parsed,
  raw,
}: {
  parsed: ParsedContent;
  raw: string;
}) {
  const content = parsed.detail || raw;

  const sections = detectSections(content);
  const hasRealSections = sections.filter(s => s.heading !== null).length >= 1;

  if (hasRealSections) {
    return (
      <div className="mt-4 expand-enter space-y-2">
        {sections.map((section, i) => (
          <div key={i}>
            {section.heading && (
              <div className="text-[14px] font-semibold text-ink mt-5 mb-2.5 flex items-center gap-2">
                <div className="w-1 h-4 rounded-full bg-gradient-to-b from-gold/60 to-gold/20" />
                {humanizeBullet(section.heading)}
              </div>
            )}
            <SectionContent lines={section.lines} parsed={parsed} />
          </div>
        ))}
      </div>
    );
  }

  const kvItems = detectKeyValueBullets(content);
  if (kvItems.length >= 2) {
    const remainingBullets = summarizeDetail(content, parsed).filter(s => {
      const sLower = s.toLowerCase();
      return !kvItems.some(kv => sLower.includes(kv.value.toLowerCase().slice(0, 20)));
    });
    return (
      <div className="mt-4 expand-enter">
        <KeyValueList items={kvItems} />
        {remainingBullets.length > 0 && (
          <div className="mt-3 space-y-2">
            {remainingBullets.slice(0, 5).map((bullet, i) => (
              <BulletCard key={i} text={bullet} index={i} />
            ))}
          </div>
        )}
      </div>
    );
  }

  const numbered = splitNumberedContent(content);
  if (numbered.length >= 2) {
    return <NumberedList items={numbered} />;
  }

  const bullets = summarizeDetail(content, parsed);
  if (bullets.length === 0) {
    return (
      <div className="mt-4 expand-enter">
        <SimpleMd text={content} humanize maxBullets={12} />
      </div>
    );
  }
  const shown = bullets.slice(0, 12);
  const remaining = bullets.length - shown.length;
  return (
    <div className="mt-4 expand-enter">
      <div className="space-y-2">
        {shown.map((bullet, i) => (
          <BulletCard key={i} text={bullet} index={i} />
        ))}
        {remaining > 0 && (
          <span className="text-[13px] text-ink-faint ml-5 block mt-2">
            +{remaining} more
          </span>
        )}
      </div>
    </div>
  );
}

function renderInlineWithPills(text: string): (string | React.ReactElement)[] {
  const parts: (string | React.ReactElement)[] = [];
  const regex = /(\*\*(.+?)\*\*)|(`([^`]+)`)|(\d+(?:\.\d+)?%)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let keyIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(<strong key={`b${keyIdx++}`} className="text-ink font-medium">{match[2]}</strong>);
    } else if (match[4]) {
      parts.push(
        <code key={`c${keyIdx++}`} className="text-[13px] bg-surface-elevated text-ink-secondary px-1 py-0.5 rounded border border-edge-subtle font-mono">
          {match[4]}
        </code>
      );
    } else if (match[5]) {
      const val = parseFloat(match[5]);
      const pillColor = val >= 80 ? 'bg-type-lesson/15 text-type-lesson'
        : val >= 60 ? 'bg-type-reminder/15 text-type-reminder'
        : 'bg-type-error/15 text-type-error';
      parts.push(
        <span key={`p${keyIdx++}`} className={`metric-pill text-[12px] font-semibold px-1.5 py-0.5 rounded-full ${pillColor}`}>
          {match[5]}
        </span>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

function ReportBody({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: React.ReactElement[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      elements.push(<div key={i} className="h-2" />);
      continue;
    }

    if (trimmed.startsWith('>')) {
      const quoteText = trimmed.replace(/^>\s*/, '');
      elements.push(
        <blockquote key={i} className="pl-4 border-l-2 border-type-decision/40 my-2 text-[15px] text-ink leading-relaxed italic">
          {renderInline(humanizeBullet(quoteText))}
        </blockquote>
      );
      continue;
    }

    if (trimmed.startsWith('```')) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <pre key={i} className="bg-surface-elevated rounded-lg p-3 border border-edge-subtle text-[13px] text-ink-secondary font-mono overflow-x-auto my-2">
          {codeLines.join('\n')}
        </pre>
      );
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const bulletText = humanizeBullet(line.replace(/^\s*[-*]\s+/, ''));
      if (bulletText.trim()) {
        elements.push(
          <div key={i} className="flex gap-2 mt-0.5">
            <span className="text-ink-faint shrink-0 mt-[1px]">&ndash;</span>
            <span className="text-[15px] text-ink-secondary leading-relaxed">
              {renderInlineWithPills(bulletText)}
            </span>
          </div>
        );
      }
      continue;
    }

    elements.push(
      <p key={i} className="text-[14px] text-ink-secondary leading-relaxed my-1">
        {renderInlineWithPills(humanizeBullet(trimmed))}
      </p>
    );
  }

  return <>{elements}</>;
}

export function ReportView({ parsed, raw }: { parsed: ParsedContent; raw: string }) {
  const content = parsed.detail || raw;
  const parts = content.split(/^(##\s+.+)$/m);
  const elements: React.ReactElement[] = [];

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i].trim();
    if (!part) continue;

    if (/^##\s+/.test(part)) {
      elements.push(
        <h3 key={`h-${i}`} className="text-[15px] font-semibold text-ink mt-5 mb-2 pl-3 border-l-2 border-type-decision">
          {part.replace(/^##\s+/, '')}
        </h3>
      );
    } else {
      elements.push(
        <div key={`b-${i}`} className="report-section">
          <ReportBody text={part} />
        </div>
      );
    }
  }

  return (
    <div className="mt-3 expand-enter" style={{ maxWidth: '65ch' }}>
      {elements}
    </div>
  );
}

export function ExpandedContent({
  parsed,
  raw,
  eventType,
}: {
  parsed: ParsedContent;
  raw: string;
  eventType?: string | null;
}) {
  if (eventType === 'research_report' || eventType === 'sota_research') {
    return <ReportView parsed={parsed} raw={raw} />;
  }
  if (eventType === 'benchmark_update') {
    return <MetricsView parsed={parsed} raw={raw} />;
  }

  switch (parsed.category) {
    case "checkpoint":
      return <CheckpointView parsed={parsed} raw={raw} />;
    case "benchmark":
    case "stats":
      return <MetricsView parsed={parsed} raw={raw} />;
    case "entity":
      return <EntityView parsed={parsed} raw={raw} />;
    case "grant":
      return <GrantView parsed={parsed} raw={raw} />;
    default:
      return <DefaultView parsed={parsed} raw={raw} />;
  }
}

export function ArticleSummary({
  parsed,
  content,
  eventType,
}: {
  parsed: ParsedContent;
  content: string;
  eventType: string | null;
}) {
  const summary = generateExecutiveSummary(parsed, eventType, content);

  const isBenchmark = eventType === 'benchmark_update' || parsed.category === 'benchmark';
  const headlineMetric = isBenchmark
    ? parsed.metrics.find(m => ['overall', 'total', 'average', 'task-averaged'].includes(m.label.toLowerCase()))
    : null;

  if (!summary && !headlineMetric) return null;

  return (
    <div className="mt-0.5">
      {summary && (
        <p className="text-[15px] text-ink-secondary leading-relaxed line-clamp-2">
          {summary}
        </p>
      )}
      {headlineMetric && headlineMetric.unit === '%' && (
        <span className={`inline-block mt-1.5 text-[12px] font-semibold px-2 py-0.5 rounded-full ${
          parseFloat(headlineMetric.value) >= 80 ? 'bg-type-lesson/15 text-type-lesson' :
          parseFloat(headlineMetric.value) >= 60 ? 'bg-type-reminder/15 text-type-reminder' :
          'bg-type-error/15 text-type-error'
        }`}>
          {headlineMetric.value}{headlineMetric.unit}
        </span>
      )}
    </div>
  );
}

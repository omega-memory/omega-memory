import React from "react";
import type { MemoryInsight } from "../lib/types";

const SEVERITY_STYLES = {
  warning: {
    border: "border-l-type-error",
    icon: (
      <svg width={18} height={18} viewBox="0 0 16 16" fill="none" className="text-type-error shrink-0 mt-0.5">
        <path d="M8 1.5L14.5 13H1.5L8 1.5Z" stroke="currentColor" strokeWidth={1.5} strokeLinejoin="round" />
        <path d="M8 6.5V9" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
        <circle cx={8} cy={11} r={0.75} fill="currentColor" />
      </svg>
    ),
  },
  positive: {
    border: "border-l-type-lesson",
    icon: (
      <svg width={18} height={18} viewBox="0 0 16 16" fill="none" className="text-type-lesson shrink-0 mt-0.5">
        <circle cx={8} cy={8} r={6.5} stroke="currentColor" strokeWidth={1.5} />
        <path d="M5.5 8L7.25 9.75L10.5 6.5" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  info: {
    border: "border-l-type-decision",
    icon: (
      <svg width={18} height={18} viewBox="0 0 16 16" fill="none" className="text-type-decision shrink-0 mt-0.5">
        <circle cx={8} cy={8} r={6.5} stroke="currentColor" strokeWidth={1.5} />
        <path d="M8 7V11" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
        <circle cx={8} cy={5} r={0.75} fill="currentColor" />
      </svg>
    ),
  },
} as const;

export function InsightCard({ insight }: { insight: MemoryInsight }) {
  const style = SEVERITY_STYLES[insight.severity];
  return (
    <div className={`flex gap-2.5 p-3 bg-surface-elevated/50 rounded-lg border-l-2 ${style.border}`}>
      {style.icon}
      <div className="min-w-0">
        <div className="text-[16px] text-ink font-medium leading-relaxed">{insight.headline}</div>
        <div className="text-[16px] text-ink-secondary leading-relaxed mt-0.5">{insight.detail}</div>
        {insight.suggestion && (
          <div className="text-[14px] text-ink-tertiary italic leading-relaxed mt-1">{insight.suggestion}</div>
        )}
      </div>
    </div>
  );
}

export function AIInsights({ insights }: { insights: MemoryInsight[] }) {
  if (insights.length === 0) return null;

  return (
    <div className="space-y-2 admin-stagger">
      {insights.map((insight, i) => (
        <InsightCard key={i} insight={insight} />
      ))}
    </div>
  );
}

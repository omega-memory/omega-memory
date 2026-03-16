import React, { useState, useRef, useEffect } from "react";
import type { InsightsData } from "../lib/types";
import { TYPE_REGISTRY } from "../contentUtils";

export default function SummaryCards({
  summary,
  memory,
  content,
  period,
}: {
  summary: InsightsData["summary"];
  memory: InsightsData["memory"];
  content: InsightsData["content"] | null;
  period: number;
}) {
  const [openTooltip, setOpenTooltip] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close tooltip when clicking outside
  useEffect(() => {
    function handleOutside(e: MouseEvent | TouchEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenTooltip(null);
      }
    }
    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("touchstart", handleOutside);
    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("touchstart", handleOutside);
    };
  }, []);
  const topTypes = Object.entries(memory.byType)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const cards: {
    label: string;
    value: string;
    badge?: string;
    badgeColor?: string;
    details: string[];
  }[] = [
    {
      label: "TOTAL MEMORIES",
      value: summary.totalMemories.toLocaleString(),
      badge:
        summary.growthPct === -1
          ? "new"
          : summary.growthPct === 0
            ? "0%"
            : `${summary.growthPct > 0 ? "+" : ""}${summary.growthPct}%`,
      badgeColor:
        summary.growthPct === -1 || summary.growthPct > 0
          ? "text-type-lesson"
          : summary.growthPct < 0
            ? "text-type-error"
            : "text-ink-tertiary",
      details: [
        `${period}-day period`,
        ...(memory.totalCloud ? [`${memory.totalCloud} total in cloud`] : []),
        ...topTypes.map(([t, c]) => `${TYPE_REGISTRY[t]?.label || t}: ${c}`),
      ],
    },
    {
      label: "ACTIVE SESSIONS",
      value: summary.activeSessions.toString(),
      details: [
        `Unique session IDs in ${period}d`,
        "Each Claude Code session = 1 ID",
      ],
    },
    ...(content ? [{
      label: "PUBLISHED",
      value: summary.contentPublished.toString(),
      details: [
        `Tweets: ${content.tweets.published} published, ${content.tweets.pending} pending`,
        `LinkedIn: ${content.linkedin.published} published, ${content.linkedin.pending} pending`,
        ...(content.tweets.queueDepth > 0 ? [`Scheduled: ${content.tweets.queueDepth}`] : []),
        ...(content.tweets.failed + content.linkedin.failed > 0
          ? [`Failed: ${content.tweets.failed + content.linkedin.failed}`]
          : []),
      ],
    },
    {
      label: "SUCCESS RATE",
      value: `${summary.publishSuccessRate}%`,
      badgeColor:
        summary.publishSuccessRate >= 90
          ? "text-type-lesson"
          : summary.publishSuccessRate >= 70
            ? "text-type-reminder"
            : "text-type-error",
      details: [
        `${content.tweets.published + content.linkedin.published} published / ${content.tweets.failed + content.linkedin.failed} failed`,
        ...(content.tweets.retries + content.linkedin.retries > 0
          ? [`${content.tweets.retries + content.linkedin.retries} total retries`]
          : []),
        summary.publishSuccessRate === 100 ? "No failures" : "",
      ].filter(Boolean),
    }] : []),
  ];

  return (
    <div ref={containerRef} className="grid grid-cols-2 lg:grid-cols-4 gap-3 admin-stagger">
      {cards.map((card) => {
        const isOpen = openTooltip === card.label;
        return (
          <div
            key={card.label}
            className="group relative p-4 admin-card cursor-default transition-all duration-200"
            onClick={() => setOpenTooltip(isOpen ? null : card.label)}
            onMouseEnter={() => setOpenTooltip(card.label)}
            onMouseLeave={() => setOpenTooltip(null)}
          >
            <div className="admin-section-label mb-1">
              {card.label}
            </div>
            <div className="flex items-baseline gap-2">
              <span className={`text-[24px] font-semibold font-mono tabular-nums ${card.badgeColor && !card.badge ? card.badgeColor : "text-ink"}`}>
                {card.value}
              </span>
              {card.badge && (
                <span className={`text-[16px] font-medium ${card.badgeColor}`}>
                  {card.badge}
                </span>
              )}
            </div>
            <div className={`absolute left-0 right-0 top-full mt-1 z-50 transition-opacity duration-150 ${isOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}>
              <div className="mx-2 p-3 rounded-lg bg-surface-elevated border border-edge shadow-lg">
                {card.details.map((line, i) => (
                  <div key={i} className="text-[14px] text-ink-secondary leading-relaxed">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

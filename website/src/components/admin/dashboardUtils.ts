import type { InsightsData, DashboardData } from "./lib/types";
import type { Suggestion } from "./dashboard/SuggestionsPanel";

// Derive actionable suggestions, excluding paused projects
export function buildSuggestions(
  dashboard: DashboardData,
  insights: InsightsData,
  pausedIds: Set<string>,
): Suggestion[] {
  const suggestions: Suggestion[] = [];

  // Only suggest for active projects
  if (!pausedIds.has("omega")) {
    if (dashboard.pypi && dashboard.pypi.weeklyDownloads < dashboard.pypi.monthlyDownloads / 4) {
      suggestions.push({
        text: "PyPI downloads are slowing this week. Consider a dev.to post or Reddit thread.",
      });
    }
  }

  // Check for failing jobs (system-level, always relevant)
  const failing = (insights.schedules ?? []).filter(
    (s) => s.enabled && s.lastStatus === "error" && s.lastRunAt
  );
  if (failing.length > 0) {
    suggestions.push({
      text: `${failing.length} job${failing.length > 1 ? "s" : ""} failing: ${failing.map((f) => f.name).join(", ")}. Check scheduler.`,
    });
  }

  // Check engagement trend (only if content projects are active)
  if (!pausedIds.has("omega") && !pausedIds.has("omega-website")) {
    if (dashboard.performance?.overall?.recentTrend === "declining") {
      suggestions.push({
        text: `Engagement rate declining (${dashboard.performance.overall.avgEngagementRate.toFixed(1)}% avg). Try a different content format.`,
      });
    } else if (dashboard.performance?.overall?.recentTrend === "improving") {
      suggestions.push({
        text: `Engagement rate improving (${dashboard.performance.overall.avgEngagementRate.toFixed(1)}% avg). Current content format is working.`,
      });
    }
  }

  // Cloud sync (system-level)
  if (!insights.memory.totalCloud) {
    suggestions.push({ text: "Cloud sync has no data. Check connection." });
  }

  return suggestions.slice(0, 4);
}

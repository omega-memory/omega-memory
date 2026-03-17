// Barrel re-export: all consumers can still import from "./contentUtils"

// Types and constants — re-export everything from types
export {
  type ContentCategory,
  type ParsedContent,
  type GrantData,
  type LeaderboardEntry,
  TYPE_META,
  DEFAULT_META,
  TYPE_VERBS,
  SECTION_LABELS,
  PROJECT_NAMES,
  GRANT_STATUS_STYLES,
  TYPE_REGISTRY,
} from "./types";

// Parsing and classification
export {
  isConfidentialContent,
  parseGrantData,
  classifyContent,
  contextAwareReplace,
  humanizeBullet,
  parseContentRich,
  summarizeDetail,
  detectKeyValueBullets,
  splitNumberedContent,
  detectSections,
  extractContentStats,
  generateExecutiveSummary,
  parseLeaderboard,
} from "./parse";

// Formatting utilities
export {
  humanizeSectionHeading,
  humanProject,
  timeAgo,
  timeUntil,
  fmtNum,
  statusDot,
  humanSchedule,
  ictDate,
  ictTime,
  formatRelativeTime,
} from "./format";

// React components
export {
  renderInline,
  SimpleMd,
  SummaryBullets,
  CategoryIcon,
  DocTypeIcon,
  KeyValueList,
  NumberedList,
  SectionContent,
  GrantStatsBar,
  MetricsView,
  CheckpointView,
  EntityView,
  GrantView,
  DefaultView,
  ReportView,
  ExpandedContent,
  ArticleSummary,
} from "./components";

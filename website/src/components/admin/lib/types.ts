// Shared type definitions for admin components

export const TABS = ["dashboard", "projects", "feed", "actions", "insights", "research", "docs", "jobs", "settings", "coordination", "entities", "growth", "diagnostic"] as const;
export type Tab = typeof TABS[number];

// Tabs visible to contributors (Pro users). Owner sees all tabs.
export const CONTRIBUTOR_TABS: ReadonlySet<Tab> = new Set(["dashboard", "projects", "feed", "insights", "coordination"]);

export function getVisibleTabs(role: string | null): readonly Tab[] {
  if (role === "owner") return TABS;
  return TABS.filter((t) => CONTRIBUTOR_TABS.has(t));
}

export const TAB_LABELS: Record<Tab, string> = {
  dashboard: "Dashboard",
  projects: "Projects",
  feed: "Feed",
  actions: "Social",
  insights: "Insights",
  research: "Research",
  docs: "Docs",
  jobs: "Jobs",
  settings: "Settings",
  coordination: "Coordination",
  entities: "Entities",
  growth: "Growth",
  diagnostic: "Diagnostic",
};

/** Account filter used across all tabs. "all" = aggregate view. */
export type XAccount = "all" | "jasonsosa" | "omega_memory";

export interface HeatmapCell {
  total: number;
  types: Record<string, number>;
  projects: Record<string, number>;
}

// ─── Project Radar ───────────────────────────────────────

export interface ProjectAllocation {
  displayName: string;
  percent: number;
  delta: number;
}

export interface ProjectRadarCard {
  displayName: string;
  momentum: "up" | "steady" | "cooling" | "stalled";
  momentumDelta: number;
  sessionCount: number;
  decisionCount: number;
  taskCompletedCount: number;
  narrative: string | null;
  lastActive: string;
  daysSinceLastSession: number;
  isActive: boolean;
  alerts: string[];
  recentDecisions: { content: string; createdAt: string }[];
  blockers: { content: string; createdAt: string }[];
  velocitySeries: { week: string; sessions: number; decisions: number }[];
}

export interface HealthSubCheck {
  label: string;
  value: string;
  status?: "healthy" | "warning" | "error";
}

export interface InsightsData {
  period: number;
  summary: {
    totalMemories: number;
    growthPct: number;
    activeSessions: number;
    contentPublished: number;
    publishSuccessRate: number;
  };
  memory: {
    byDay: Record<string, number>;
    byType: Record<string, number>;
    total: number;
    totalAll: number;
    totalCloud?: number | null;
  };
  lastMemoryAt?: string;
  content: {
    tweets: ContentStats;
    linkedin: ContentStats;
  } | null;
  allocation: ProjectAllocation[];
  projects: ProjectRadarCard[];
  unscopedCount: number;
  schedules: ScheduleInfo[] | null;
  heatmap?: Record<string, HeatmapCell>;
  memoryInsights: MemoryInsight[];
  verification?: VerificationMetrics;
}

export interface ContentStats {
  pending: number;
  approved: number;
  published: number;
  failed: number;
  rejected: number;
  retries: number;
  byContentType?: Record<string, number>;
  queueDepth: number;
  nextScheduled: string | null;
}

export interface ScheduleInfo {
  id: string;
  name: string;
  label: string;
  enabled: boolean;
  lastStatus: string;
  lastRunAt: string | null;
  scheduleType: string;
  intervalSeconds?: number;
  calendarHour?: number;
  calendarMinute?: number;
  calendarWeekday?: number | null;
  overdue: boolean;
}

export interface TweetMetrics {
  id: string;
  text: string;
  createdAt: string;
  impressions: number;
  likes: number;
  retweets: number;
  replies: number;
  bookmarks: number;
}

export interface PerformanceData {
  byContentType: { contentType: string; avgEngagementRate: number; count: number }[];
  byTheme: { theme: string; avgEngagementRate: number; count: number }[];
  bySlot: { slotNumber: number; avgEngagementRate: number; count: number }[];
  topPerformers: {
    text: string;
    contentType: string;
    engagementRate: number;
    impressions: number;
    likes: number;
    retweets: number;
    replies: number;
    bookmarks: number;
  }[];
  overall: {
    totalTracked: number;
    avgEngagementRate: number;
    recentTrend: "improving" | "steady" | "declining";
  };
}

export interface DashboardData {
  github: {
    stars: number;
    forks: number;
    openIssues: number;
    pushedAt: string;
    contributors: number;
    watchers: number;
  } | null;
  pypi: {
    version: string;
    monthlyDownloads: number;
    weeklyDownloads: number;
  } | null;
  twitter: {
    followers: number;
    following: number;
    tweetCount: number;
    recentTweets: TweetMetrics[];
    accounts?: {
      jasonsosa: { handle: string; followers: number; following: number; tweetCount: number } | null;
      omega_memory: { handle: string; followers: number; following: number; tweetCount: number } | null;
    };
  } | null;
  outreach: {
    pending: number;
    approved: number;
    sent: number;
    failed: number;
    recentSent: { handle: string; status: string; sentAt: string }[];
  };
  grants: {
    id: string;
    content: string;
    type: string;
    createdAt: string;
    inferredStage: string;
    project: string;
  }[];
  performance: PerformanceData | null;
  downloads: {
    total: number;
    macos: number;
    windows: number;
    thisWeek: number;
  } | null;
}

// ─── Workflow Changes ────────────────────────────────────

export interface WorkflowMetricChange {
  label: string;
  thisWeek: number;
  lastWeek: number;
  direction: "up" | "down" | "flat";
  magnitude: number;
  unit: string;
  context?: string;
}

export interface CoachingItem {
  id: string;
  text: string;
  category: string;
  impact: string;
  firstShown: string;
  trend: "improving" | "worsening" | "flat";
  graduated: boolean;
  graduatedAt?: string;
  evidence?: string;
}

export interface WorkPattern {
  dimension: string;
  dimensionLabel: string;
  description: string;
  confidence: number;
  evidenceCount: number;
  sessionCount: number;
}

export interface WorkflowChangesData {
  changes: WorkflowMetricChange[];
  coaching: CoachingItem[];
  strengths: WorkPattern[];
  hasData: boolean;
  sessionCount: number;
}

export interface ActionItem {
  severity: "critical" | "warning" | "info";
  label: string;
  detail: string;
  action: string;
  tab?: Tab;
  href?: string;
}

// ─── Memory Insights ─────────────────────────────────────────

export type InsightSeverity = "warning" | "positive" | "info";

export interface MemoryInsight {
  severity: InsightSeverity;
  headline: string;
  detail: string;
  suggestion?: string;
}

// ─── Verification Metrics ────────────────────────────────

export interface VerificationLayer {
  label: string;
  rho: number;
  verified: number;
  total: number;
}

export interface VerificationMetrics {
  overallRho: number;
  layers: VerificationLayer[];
}

// -- Coordination Dashboard --------------------------------------------------

export interface CoordinationSession {
  session_id: string;
  project: string;
  status: string;
  task: string;
  last_heartbeat: string;
  started_at: string;
  pid?: number | null;
  capabilities?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface CoordinationFileClaim {
  file_path: string;
  session_id: string;
  task: string;
  claimed_at: string;
  last_activity?: string | null;
}

export interface CoordinationAuditEntry {
  id: number;
  tool_name: string;
  result_summary: string | null;
  created_at: string;
  call_index: number | null;
  result_status: string;
  input_size: number | null;
  latency_ms: number | null;
}

export interface CoordinationTask {
  id: number;
  local_id: number | null;
  title: string;
  description: string | null;
  status: string;
  priority: number;
  progress: number;
  created_at: string;
  claimed_at: string | null;
  completed_at: string | null;
  result: string | null;
}

export interface CoordinationFileRead {
  session_id: string;
  file_path: string;
  first_read_at: string;
  read_count: number;
}

export interface CoordinationMessage {
  id: number;
  from_session: string;
  to_session: string | null;
  project: string | null;
  msg_type: string;
  context_id: string | null;
  subject: string;
  body: string | null;
  created_at: string;
  read_at: string | null;
  priority?: string | null;
  batch_id?: string | null;
  delivered_at?: string | null;
}

export interface CoordinationHandoff {
  id: number;
  session_id: string;
  project: string | null;
  completed_tasks: string;
  blocked_items: string;
  key_context: string | null;
  next_steps: string;
  files_modified: string;
  decisions_made: string;
  git_branch: string | null;
  git_dirty_files: string;
  created_at: string;
  read_by: string;
}

export interface CoordinationGitEvent {
  id: number;
  session_id: string | null;
  project: string;
  event_type: string;
  commit_hash: string | null;
  branch: string | null;
  message: string | null;
  created_at: string;
}

export interface CoordinationIntent {
  id: number;
  session_id: string;
  intent_type: string;
  description: string;
  target_files: string | null;
  target_branch: string | null;
  findings: string | null;
  created_at: string;
  expires_at: string | null;
}

export interface CoordinationDecision {
  id: number;
  domain: string;
  project: string;
  decision: string;
  rationale: string | null;
  decided_by: string;
  goal_id: number | null;
  status: string;
  superseded_by: number | null;
  created_at: string;
  superseded_at: string | null;
}

export interface CoordinationMetric {
  id: number;
  metric_name: string;
  metric_value: number;
  session_id: string | null;
  project: string | null;
  metadata: string | null;
  created_at: string;
}

export interface CoordinationData {
  sessions: CoordinationSession[];
  file_claims: CoordinationFileClaim[];
  file_reads: CoordinationFileRead[];
  messages: CoordinationMessage[];
  handoffs: CoordinationHandoff[];
  intents: CoordinationIntent[];
  decisions: CoordinationDecision[];
  tasks: CoordinationTask[];
  git_events: CoordinationGitEvent[];
}

// ─── Entity Admin ──────────────────────────────────────────

export interface EntityListItem {
  id: string;
  name: string;
  entityType: string;
  jurisdiction: string | null;
  status: string;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
  memoryCount: number;
  relationshipCount: number;
  projects: string[];
  lastSeen: string | null;
}

export interface EntityTypeCount {
  type: string;
  count: number;
}

export interface EntityProjectCount {
  project: string;
  count: number;
}

export interface EntityListData {
  entities: EntityListItem[];
  total: number;
  totalUserEntities: number;
  filtered: number;
  entityTypes: EntityTypeCount[];
  projects: EntityProjectCount[];
}

// ─── Diagnostic ─────────────────────────────────────────────

export interface DiagnosticRecommendation {
  severity: "critical" | "warning" | "info";
  message: string;
  action?: string;
}

export interface DiagnosticHealthCheck {
  label: string;
  status: "healthy" | "warning" | "error";
  detail: string;
  sub_checks: HealthSubCheck[];
}

export interface DiagnosticAgentSession {
  session_id: string;
  client: string;
  transport: string;
  project: string;
  status: "active" | "idle" | "ended";
  last_seen: string;
  tool_calls: number;
}

export interface DiagnosticData {
  verdict: "healthy" | "underused" | "idle";
  memory_health: {
    total: number;
    utilization_pct: number;
    velocity_7d: { event_type: string; count: number }[];
    velocity_total_7d: number;
    dead_memories: number;
    dead_pct: number;
    dead_by_type: { event_type: string; count: number }[];
    access_buckets: { never: number; low: number; medium: number; high: number };
  };
  system_health: {
    checks: DiagnosticHealthCheck[];
    overall: "healthy" | "degraded" | "down";
  };
  quality: {
    avg_priority: number;
    priority_distribution: { priority: number; count: number }[];
    by_project: { project: string; count: number }[];
    total_with_tags: number;
    total_with_project: number;
    tags_pct: number;
    project_pct: number;
  };
  tool_usage: {
    top_tools: { tool: string; calls: number }[];
    omega_tools: { tool: string; calls: number }[];
    total_calls: number;
    omega_calls: number;
    omega_per_session: number;
  };
  latency: {
    avg_ms: number;
    p95_ms: number;
    by_tool: { tool: string; avg_ms: number; p95_ms: number; calls: number }[];
  };
  sessions: { total: number; week: number; month: number; period: number };
  llm_costs: {
    total_usd: number;
    total_input_tokens: number;
    total_output_tokens: number;
    total_sessions: number;
    by_model: { model: string; cost: number; calls: number }[];
  };
  recommendations: DiagnosticRecommendation[];
  period_days: number;
}

// ─── Alert Context (xyOps-inspired) ────────────────────────

export type AlertType = "failing_job" | "failed_post" | "cloud_sync_gap" | "engagement_declining" | "memory_spike" | "coordination_conflict" | "overdue_job" | "cloud_sync_empty";

export interface AlertContextBase {
  type: AlertType;
  title: string;
  severity: "critical" | "warning" | "info";
  timestamp: string;
}

export interface FailingJobContext extends AlertContextBase {
  type: "failing_job";
  detail: {
    jobName: string;
    jobLabel: string;
    lastError: string | null;
    recentRuns: { status: string; startedAt: string; durationMs: number | null; error: string | null }[];
    scheduleType: string;
    intervalSeconds?: number;
  };
}

export interface FailedPostContext extends AlertContextBase {
  type: "failed_post";
  detail: {
    failedCount: number;
    recentFailed: { content: string; reason: string | null; createdAt: string; account: string }[];
    recentSuccessful: { content: string; publishedAt: string; account: string }[];
  };
}

export interface CloudSyncGapContext extends AlertContextBase {
  type: "cloud_sync_gap";
  detail: {
    localCount: number;
    cloudCount: number;
    gapPct: number;
    lastSyncAt: string | null;
    unsyncedCount: number;
  };
}

export interface EngagementDecliningContext extends AlertContextBase {
  type: "engagement_declining";
  detail: {
    currentRate: number;
    previousRate: number;
    recentPosts: { content: string; engagementRate: number; publishedAt: string }[];
  };
}

export interface MemorySpikeContext extends AlertContextBase {
  type: "memory_spike";
  detail: {
    recentMemories: { content: string; memoryType: string; agentId: string | null; createdAt: string }[];
    totalInLastHour: number;
  };
}

export interface CoordinationConflictContext extends AlertContextBase {
  type: "coordination_conflict";
  detail: {
    conflicts: { filePath: string; sessions: { sessionId: string; project: string; intent: string | null }[] }[];
  };
}

export type AlertContext = FailingJobContext | FailedPostContext | CloudSyncGapContext | EngagementDecliningContext | MemorySpikeContext | CoordinationConflictContext | AlertContextBase;

// ─── Alert Actions ──────────────────────────────────────────

export type AlertActionType = "retry_job" | "force_sync" | "dismiss" | "snooze" | "navigate" | "requeue_post";

export interface AlertAction {
  type: AlertActionType;
  label: string;
  /** For navigate: tab to switch to */
  tab?: Tab;
  /** For retry_job: schedule label */
  jobLabel?: string;
  /** Severity styling */
  variant?: "primary" | "secondary" | "danger";
}

// ─── Hook Pipeline ──────────────────────────────────────────

export interface HookExecution {
  hookType: string;
  status: "success" | "error" | "skipped";
  durationMs: number;
  timestamp: string;
  payload?: string | null;
  output?: string | null;
  error?: string | null;
}

export interface HookNode {
  id: string;
  label: string;
  status: "success" | "error" | "inactive";
  executionCount: number;
  avgDurationMs: number;
  lastExecuted: string | null;
  recentExecutions: HookExecution[];
}

export interface HookEdge {
  from: string;
  to: string;
}

export interface HookPipelineData {
  nodes: HookNode[];
  edges: HookEdge[];
  periodDays: number;
}

// ─── Incident Timeline ──────────────────────────────────────

export type TimelineEventSource = "coordination" | "memory" | "job" | "hook" | "git";

export interface TimelineEvent {
  id: string;
  source: TimelineEventSource;
  eventType: string;
  title: string;
  detail: string | null;
  timestamp: string;
  agentId: string | null;
  project: string | null;
  /** For drill-down navigation */
  linkedTab?: Tab;
  linkedId?: string;
}

export interface TimelineData {
  events: TimelineEvent[];
  total: number;
  hasMore: boolean;
}

// ─── Sidebar Groups ─────────────────────────────────────────

export interface SidebarGroup {
  key: string;
  label: string;
  tabs: Tab[];
}

export const SIDEBAR_GROUPS: SidebarGroup[] = [
  { key: "observe", label: "Observe", tabs: ["dashboard", "feed", "insights"] },
  { key: "act", label: "Act", tabs: ["actions", "research", "docs"] },
  { key: "operate", label: "Operate", tabs: ["projects", "coordination", "jobs", "diagnostic"] },
  { key: "configure", label: "Configure", tabs: ["entities", "growth", "settings"] },
];

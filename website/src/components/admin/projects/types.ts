import type {
  CoordinationSession,
  CoordinationHandoff,
  CoordinationDecision,
  CoordinationGitEvent,
} from "../lib/types";

export interface ProjectTimelineEntry {
  session: CoordinationSession;
  handoff: CoordinationHandoff | null;
  decisions: CoordinationDecision[];
  gitEvents: CoordinationGitEvent[];
  durationMinutes: number;
}

// ─── Project Overview (redesign) ────────────────────────────

export type ProjectHealth = "healthy" | "blocked" | "stale" | "attention";

export interface TaskItem {
  id: number;
  title: string;
  status: string;
  priority: number;
  progress: number;
}

export interface DecisionItem {
  id: number;
  domain: string;
  decision: string;
  rationale: string | null;
  createdAt: string;
}

export interface FileClaimItem {
  filePath: string;
  sessionId: string;
  claimedAt: string;
}

export interface ActivityItem {
  type: "session" | "commit" | "decision" | "task";
  summary: string;
  timestamp: string;
  detail?: string | null;
  filesChanged?: string[];
  branch?: string | null;
}

export interface ProjectOverview {
  id: string;
  name: string;
  category: string;
  health: ProjectHealth;
  lastActive: string | null;
  hasActiveSession: boolean;
  summary: string | null;
  gitBranch: string | null;
  sessionCount7d: number;
  sessionCount30d: number;
  commitCount30d: number;
  decisionCount30d: number;
  launchProgress: number;
  tasks: {
    pending: number;
    inProgress: number;
    completed: number;
    failed: number;
    blocked: number;
    items: TaskItem[];
  };
  sparkline: number[];
  latestHandoff: {
    keyContext: string | null;
    blockedItems: string[];
    nextSteps: string[];
    gitBranch: string | null;
  } | null;
  decisions: DecisionItem[];
  fileClaims: FileClaimItem[];
  activity: ActivityItem[];
}

export interface AttentionItem {
  projectId: string;
  projectName: string;
  severity: "red" | "amber";
  reason: string;
}

export interface ProjectOverviewResponse {
  projects: ProjectOverview[];
  needsAttention: AttentionItem[];
  generatedAt: string;
}

// ─── Architecture Detail View ──────────────────────────────

export interface ArchitectureModule {
  path: string;
  fileCount: number;
  recentCommits: number;
  health: "active" | "recent" | "dormant";
}

export interface ArchitectureEdge {
  from: string;
  to: string;
  weight: number;
}

export interface SessionHeatmapEntry {
  date: string;
  hour: number;
  count: number;
}

export interface ArchitectureResponse {
  modules: ArchitectureModule[];
  edges: ArchitectureEdge[];
  mermaidSyntax: string;
  sessionHeatmap: SessionHeatmapEntry[];
}

export type BoardColumn = "blocked" | "active-now" | "this-week" | "steady" | "parked";

export type CurationStatus = "starred" | "dismissed" | "archived" | null;

export interface ResearchFinding {
  id: string;
  content: string;
  created_at: string;
  source_url: string | null;
  source_type: string;
  technique: string | null;
  impact_score: number | null;
  feasibility_score: number | null;
  priority_score: number;
  subsystem: string | null;
  source: string | null;
  domain: string | null;
  curation_status: CurationStatus;
}

export interface ResearchPlan {
  id: string;
  title: string;
  content: string;
  status: "draft" | "ready" | "implemented";
  subsystem: string | null;
  finding_id: string | null;
  created_at: string;
}

export interface ResearchStats {
  total_findings: number;
  scanner_findings: number;
  high_priority: number;
  plan_count: number;
  subsystems: string[];
}

export interface ScanHistoryEntry {
  scan_date: string;
  raw_findings: number;
  kept: number;
  discarded: number;
  duration_ms: number | null;
  findings_summary: Array<{
    title: string;
    technique: string;
    priority_score: number;
    subsystem: string | null;
    status: string;
  }>;
}

export interface ResearchData {
  findings: ResearchFinding[];
  plans: ResearchPlan[];
  total: number;
  stats: ResearchStats;
  scan_history: ScanHistoryEntry[];
}

export interface ScanReport {
  scan_date: string;
  sources_checked: number;
  raw_findings: number;
  kept: number;
  discarded: number;
  duration_ms: number;
  findings: Array<{
    title: string;
    technique: string;
    priority_score: number;
    status: string;
    subsystem: string | null;
  }>;
}

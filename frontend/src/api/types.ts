export interface HealthResponse {
  status: "ok";
  database: "ok";
}

export interface SourceSummary {
  id: string;
  type: "manual";
  title: string;
  author: string | null;
  processing_status: "ready";
  created_at: string;
  updated_at: string;
}

export interface SourceDetail extends SourceSummary {
  raw_text: string;
}

export interface DashboardResponse {
  source_count: number;
  recent_sources: SourceDetail[];
}

export interface ManualSourceInput {
  text: string;
  title?: string | null;
  author?: string | null;
}

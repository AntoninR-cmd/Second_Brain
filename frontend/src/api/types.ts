export interface HealthResponse {
  status: "ok";
  database: "ok";
}

export type SourceType = "manual" | "srt" | "txt";
export type ProcessingStatus = "ready";
export type AnalysisStatus =
  | "not_analyzed"
  | "queued"
  | "processing"
  | "analyzed"
  | "error";

export interface SourceSummary {
  id: string;
  type: SourceType;
  title: string;
  author: string | null;
  original_filename: string | null;
  processing_status: ProcessingStatus;
  analysis_status: AnalysisStatus;
  created_at: string;
  updated_at: string;
}

export interface SourceDetail extends SourceSummary {
  raw_text: string;
  original_file_path: string | null;
  file_sha256: string | null;
  segment_count: number;
  summary: string | null;
  analysis_error: string | null;
  analysis_started_at: string | null;
  analysis_completed_at: string | null;
  knowledge_count: number;
}

export interface SourceListResponse {
  items: SourceSummary[];
  next_cursor: string | null;
}

export interface SourceSegment {
  id: string;
  source_id: string;
  index: number;
  text: string;
  start_ms: number | null;
  end_ms: number | null;
}

export interface SourceSegmentListResponse {
  items: SourceSegment[];
  next_cursor: number | null;
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

export interface FileSourceInput {
  file: File;
  title?: string | null;
  author?: string | null;
}

export interface OllamaReadiness {
  available: boolean;
  base_url: string;
  configured_model: string;
  model_available: boolean;
  error: string | null;
}

export interface SystemReadinessResponse {
  status: "ready" | "degraded";
  database: "ok";
  ollama: OllamaReadiness;
}

export type ProcessingJobStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed";

export interface AnalysisJob {
  id: string;
  source_id: string;
  kind: "analyze_source";
  status: ProcessingJobStatus;
  stage: string | null;
  progress_current: number;
  progress_total: number;
  progress_percent: number;
  progress_message: string | null;
  error_message: string | null;
  error_code?: string | null;
  error_type?: string | null;
  error_detail?: string | null;
  error_stage?: string | null;
  error_passage_id?: string | null;
  error_passage_index?: number | null;
  error_attempt?: number | null;
  error_call_type?: string | null;
  /** Temporary aliases accepted while older Phase 3 APIs are upgraded. */
  failed_passage_id?: string | null;
  failed_passage_index?: number | null;
  failed_attempt?: number | null;
  failed_call_type?: string | null;
  attempt_count: number;
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  heartbeat_at?: string | null;
  is_stale?: boolean;
  started_at: string | null;
  finished_at: string | null;
}

export interface KnowledgeNodeSummary {
  id: string;
  source_id: string;
  title: string;
  content: string;
  tags: string[];
  evidence_count: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeNodeListResponse {
  items: KnowledgeNodeSummary[];
  next_cursor: string | null;
}

export interface KnowledgeSourceReference {
  id: string;
  type: SourceType;
  title: string;
  author: string | null;
  original_filename: string | null;
  original_file_path: string | null;
}

export interface KnowledgeEvidence {
  id: string;
  passage_id: string;
  passage_index: number;
  original_excerpt: string;
  start_ms: number | null;
  end_ms: number | null;
  first_segment_index: number | null;
  last_segment_index: number | null;
  char_start: number | null;
  char_end: number | null;
}

export interface KnowledgeNodeDetail extends KnowledgeNodeSummary {
  source: KnowledgeSourceReference;
  evidences: KnowledgeEvidence[];
}

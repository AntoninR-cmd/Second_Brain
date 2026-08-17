export interface HealthResponse {
  status: "ok";
  database: "ok";
}

export type SourceType = "manual" | "srt" | "txt";
export type ProcessingStatus = "ready";

export interface SourceSummary {
  id: string;
  type: SourceType;
  title: string;
  author: string | null;
  original_filename: string | null;
  processing_status: ProcessingStatus;
  created_at: string;
  updated_at: string;
}

export interface SourceDetail extends SourceSummary {
  raw_text: string;
  original_file_path: string | null;
  file_sha256: string | null;
  segment_count: number;
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

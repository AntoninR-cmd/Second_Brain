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

export type VectorIndexState =
  | "empty"
  | "not_built"
  | "building"
  | "ready"
  | "stale"
  | "incompatible"
  | "unavailable"
  | "corrupt";

export interface EmbeddingReadiness {
  ollama_available: boolean;
  configured_model: string;
  model_available: boolean;
  error: string | null;
}

export interface EmbeddingProfile {
  id: string;
  model_name: string;
  model_digest: string | null;
  dimensions: number | null;
  distance: "cosine";
  semantic_text_version: string;
  logical_generation: number;
  status: "building" | "active" | "retired" | "failed";
  error_message: string | null;
}

export interface VectorJob {
  id: string;
  kind: "index_knowledge" | "rebuild_vector_index";
  status: ProcessingJobStatus;
  stage: string | null;
  progress_current: number;
  progress_total: number;
  progress_percent: number;
  progress_message: string | null;
  error_message: string | null;
  error_code: string | null;
  error_type: string | null;
  error_detail: string | null;
  attempt_count: number;
  embedding_batch_count: number;
  embedding_item_count: number;
  embedding_duration_ms: number;
  embedding_total_duration_ns: number;
  embedding_prompt_eval_count: number;
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  is_stale: boolean;
  started_at: string | null;
  finished_at: string | null;
}

export interface VectorIndexStatus {
  state: VectorIndexState;
  configured_model: string;
  embedding: EmbeddingReadiness;
  total_nodes: number;
  indexed_nodes: number;
  pending_or_stale_nodes: number;
  failed_nodes: number;
  orphan_points: number;
  active_profile: EmbeddingProfile | null;
  active_job: VectorJob | null;
  latest_job: VectorJob | null;
  error: string | null;
}

export type BrainState =
  | "empty"
  | "not_built"
  | "building"
  | "ready"
  | "stale"
  | "error"
  | "vector_index_required"
  | "unavailable";

export type BrainProfileStatus = "building" | "ready" | "stale" | "error";

export interface BrainSimilarityStats {
  minimum: number | null;
  mean: number | null;
  median: number | null;
  maximum: number | null;
}

export interface BrainClusterSizeStats {
  minimum: number | null;
  mean: number | null;
  maximum: number | null;
}

export interface BrainProfile {
  id: string;
  logical_generation: number;
  status: BrainProfileStatus;
  embedding_profile_id: string | null;
  embedding_provider: string;
  embedding_model_name: string;
  embedding_model_digest: string | null;
  embedding_dimensions: number;
  embedding_semantic_text_version: string;
  embedding_logical_generation: number;
  algorithm_version: string;
  knowledge_node_count: number;
  cluster_count: number;
  edge_count: number;
  unassigned_node_count: number;
  cluster_counts_by_level: Record<string, number>;
  similarity: BrainSimilarityStats;
  cluster_sizes: BrainClusterSizeStats;
  relations_duration_ms: number;
  clustering_duration_ms: number;
  umap_duration_ms: number;
  labeling_duration_ms: number;
  total_duration_ms: number;
  label_strategy: "deterministic" | "ollama" | "mixed";
  label_model_name: string | null;
  label_model_digest: string | null;
  created_at: string;
  completed_at: string | null;
  activated_at: string | null;
  error_message: string | null;
}

export interface BrainJob {
  id: string;
  brain_profile_id: string | null;
  kind: "build_brain" | "relabel_brain";
  status: ProcessingJobStatus;
  stage: string | null;
  progress_current: number;
  progress_total: number;
  progress_percent: number;
  progress_message: string | null;
  error_message: string | null;
  error_code: string | null;
  error_type: string | null;
  error_detail: string | null;
  error_stage: string | null;
  attempt_count: number;
  llm_call_count: number;
  llm_retry_count: number;
  llm_duration_ms: number;
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  is_stale: boolean;
  started_at: string | null;
  finished_at: string | null;
}

export interface BrainStatus {
  state: BrainState;
  active_profile: BrainProfile | null;
  building_profile: BrainProfile | null;
  active_job: BrainJob | null;
  latest_job: BrainJob | null;
  stale_reasons: string[];
  can_rebuild: boolean;
  can_relabel: boolean;
  error: string | null;
}

export interface BrainCluster {
  id: string;
  parent_id: string | null;
  level: number;
  label: string;
  description: string | null;
  label_source: "deterministic" | "ollama";
  member_count: number;
  representative_knowledge_node_ids: string[];
  x: number;
  y: number;
  child_count: number;
}

export interface BrainKnowledgeNode {
  id: string;
  cluster_id: string | null;
  title: string;
  tags: string[];
  source_id: string;
  source_title: string;
  x: number;
  y: number;
  is_unassigned: boolean;
  href: string;
}

export interface BrainClusterDetail extends BrainCluster {
  children: BrainCluster[];
  knowledge_nodes: BrainKnowledgeNode[];
}

export interface BrainGraphNode {
  id: string;
  kind: "cluster" | "knowledge";
  label: string;
  x: number;
  y: number;
  size: number;
  cluster_id: string | null;
  knowledge_node_id: string | null;
  source_id: string | null;
  tags: string[];
  href: string | null;
}

export interface BrainGraphEdge {
  source: string;
  target: string;
  score: number;
  relation_count: number;
}

export interface BrainGraphResponse {
  profile_id: string;
  level: number;
  parent_cluster_id: string | null;
  nodes: BrainGraphNode[];
  edges: BrainGraphEdge[];
  truncated: boolean;
}

export interface BrainSearchAncestor {
  id: string;
  label: string;
  level: number;
}

export interface BrainSearchResult {
  kind: "cluster" | "knowledge";
  target_id: string;
  label: string;
  level: number | null;
  cluster_id: string | null;
  x: number;
  y: number;
  member_count: number | null;
  tags: string[];
  source_id: string | null;
  source_title: string | null;
  href: string | null;
  ancestors: BrainSearchAncestor[];
}

export interface BrainSearchResponse {
  profile_id: string;
  query: string;
  items: BrainSearchResult[];
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

export interface SemanticSearchInput {
  query: string;
  top_k?: number;
}

export interface SemanticSearchResult {
  score: number;
  href: string;
  knowledge_node: KnowledgeNodeSummary;
  source: KnowledgeSourceReference;
  evidences: KnowledgeEvidence[];
}

export interface SemanticSearchResponse {
  query: string;
  items: SemanticSearchResult[];
  profile: {
    model_name: string;
    dimensions: number;
    distance: "cosine";
  } | null;
}

export type RagMode = "brain_only" | "brain_plus_model";

export interface RagAnswerInput {
  question: string;
  mode: RagMode;
  top_k?: number;
}

export interface RagKnowledge extends SemanticSearchResult {
  context_id: string | null;
  provided_to_model: boolean;
  used: boolean;
}

export interface RagTimings {
  readiness_ms: number;
  embedding_ms: number;
  qdrant_ms: number;
  retrieval_sqlite_ms: number;
  context_build_ms: number;
  generation_ms: number;
  provenance_validation_ms: number;
  total_ms: number;
  prompt_eval_count: number | null;
  eval_count: number | null;
}

export interface RagAnswerResponse {
  request_id: string;
  question: string;
  mode: RagMode;
  answer: string;
  model_additions: string | null;
  insufficient_context: boolean;
  generation_model: string;
  retrieved_knowledge: RagKnowledge[];
  used_knowledge: RagKnowledge[];
  timings: RagTimings;
  citation_format: string;
}

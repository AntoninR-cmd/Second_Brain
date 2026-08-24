import type {
  AnalysisJob,
  DashboardResponse,
  FileSourceInput,
  HealthResponse,
  KnowledgeNodeDetail,
  KnowledgeNodeListResponse,
  ManualSourceInput,
  RagAnswerInput,
  RagAnswerResponse,
  SemanticSearchInput,
  SemanticSearchResponse,
  SourceDetail,
  SourceListResponse,
  SourceSegmentListResponse,
  SystemReadinessResponse,
  VectorIndexStatus,
  VectorJob,
} from "./types";

const API_PREFIX = "/api/v1";

interface ValidationIssue {
  msg?: string;
}

interface ApiErrorBody {
  detail?: string | ValidationIssue[];
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function errorMessage(body: ApiErrorBody | null, status: number): string {
  if (typeof body?.detail === "string" && body.detail.trim()) {
    return body.detail;
  }

  if (Array.isArray(body?.detail)) {
    const messages = body.detail
      .map((issue) => issue.msg)
      .filter((message): message is string => Boolean(message));

    if (messages.length > 0) {
      return messages.join(" ");
    }
  }

  return `La requête a échoué (erreur ${status}).`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null;
    throw new ApiError(errorMessage(body, response.status), response.status);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/system/health");
}

export function getSystemReadiness(): Promise<SystemReadinessResponse> {
  return request<SystemReadinessResponse>("/system/readiness");
}

export function getVectorIndexStatus(): Promise<VectorIndexStatus> {
  return request<VectorIndexStatus>("/vector-index/status");
}

export function indexKnowledgeNodes(): Promise<VectorJob> {
  return request<VectorJob>("/vector-index/index", { method: "POST" });
}

export function rebuildVectorIndex(): Promise<VectorJob> {
  return request<VectorJob>("/vector-index/rebuild", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ confirm: true }),
  });
}

export function getVectorJob(jobId: string): Promise<VectorJob> {
  return request<VectorJob>(
    `/vector-index/jobs/${encodeURIComponent(jobId)}`,
  );
}

export function searchSemantically(
  input: SemanticSearchInput,
): Promise<SemanticSearchResponse> {
  return request<SemanticSearchResponse>("/search/semantic", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });
}

export function answerWithRag(
  input: RagAnswerInput,
): Promise<RagAnswerResponse> {
  return request<RagAnswerResponse>("/rag/answer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });
}

export function getDashboard(): Promise<DashboardResponse> {
  return request<DashboardResponse>("/dashboard");
}

export function createManualSource(
  input: ManualSourceInput,
): Promise<SourceDetail> {
  return request<SourceDetail>("/sources/manual", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });
}

export function uploadSource(input: FileSourceInput): Promise<SourceDetail> {
  const formData = new FormData();
  formData.append("file", input.file);

  if (input.title) {
    formData.append("title", input.title);
  }

  if (input.author) {
    formData.append("author", input.author);
  }

  return request<SourceDetail>("/sources/upload", {
    method: "POST",
    body: formData,
  });
}

export function getSources(cursor?: string | null): Promise<SourceListResponse> {
  const parameters = new URLSearchParams({ limit: "50" });

  if (cursor) {
    parameters.set("cursor", cursor);
  }

  return request<SourceListResponse>(`/sources?${parameters.toString()}`);
}

export function getSource(sourceId: string): Promise<SourceDetail> {
  return request<SourceDetail>(`/sources/${encodeURIComponent(sourceId)}`);
}

export function getSourceSegments(
  sourceId: string,
  cursor?: number | null,
): Promise<SourceSegmentListResponse> {
  const parameters = new URLSearchParams({ limit: "100" });

  if (cursor !== undefined && cursor !== null) {
    parameters.set("cursor", cursor.toString());
  }

  return request<SourceSegmentListResponse>(
    `/sources/${encodeURIComponent(sourceId)}/segments?${parameters.toString()}`,
  );
}

export function analyzeSource(sourceId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(
    `/sources/${encodeURIComponent(sourceId)}/analyze`,
    { method: "POST" },
  );
}

export function getSourceAnalysis(sourceId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(
    `/sources/${encodeURIComponent(sourceId)}/analysis`,
  );
}

export function getSourceKnowledgeNodes(
  sourceId: string,
  cursor?: string | null,
): Promise<KnowledgeNodeListResponse> {
  const parameters = new URLSearchParams({ limit: "100" });

  if (cursor) {
    parameters.set("cursor", cursor);
  }

  return request<KnowledgeNodeListResponse>(
    `/sources/${encodeURIComponent(sourceId)}/nodes?${parameters.toString()}`,
  );
}

export function getKnowledgeNode(
  nodeId: string,
): Promise<KnowledgeNodeDetail> {
  return request<KnowledgeNodeDetail>(`/nodes/${encodeURIComponent(nodeId)}`);
}

export function getReadableError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof TypeError) {
    return "Impossible de joindre l’API. Vérifiez que le backend est démarré.";
  }

  return "Une erreur inattendue est survenue. Réessayez.";
}

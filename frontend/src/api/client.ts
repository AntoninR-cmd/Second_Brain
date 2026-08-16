import type {
  DashboardResponse,
  HealthResponse,
  ManualSourceInput,
  SourceDetail,
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

export function getReadableError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof TypeError) {
    return "Impossible de joindre l’API. Vérifiez que le backend est démarré.";
  }

  return "Une erreur inattendue est survenue. Réessayez.";
}

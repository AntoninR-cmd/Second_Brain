// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  AnalysisJob,
  SourceDetail,
  SourceSegment,
} from "../../api/types";
import { SourceDetailPage } from "../SourceDetailPage";

const api = vi.hoisted(() => ({
  analyzeSource: vi.fn(),
  getSource: vi.fn(),
  getSourceAnalysis: vi.fn(),
  getSourceKnowledgeNodes: vi.fn(),
  getSourceSegments: vi.fn(),
}));

vi.mock("../../api/client", () => ({
  ...api,
  getReadableError: (error: unknown) =>
    error instanceof Error ? error.message : "Erreur inconnue",
}));

function source(overrides: Partial<SourceDetail> = {}): SourceDetail {
  return {
    id: "source-id",
    type: "pdf",
    title: "Guide pratique",
    author: "Ada Exemple",
    original_filename: "guide.pdf",
    processing_status: "ready",
    analysis_status: "not_analyzed",
    created_at: "2026-09-08T10:00:00Z",
    updated_at: "2026-09-08T10:00:00Z",
    raw_text: "Première page\n\nDeuxième page",
    original_file_path: "originals/source-id/original.pdf",
    file_sha256: "hash",
    segment_count: 2,
    page_count: 2,
    chapter_count: null,
    language: "fr",
    processing_error: null,
    summary: null,
    analysis_error: null,
    analysis_started_at: null,
    analysis_completed_at: null,
    knowledge_count: 0,
    ...overrides,
  };
}

function segment(overrides: Partial<SourceSegment> = {}): SourceSegment {
  return {
    id: "segment-0",
    source_id: "source-id",
    index: 0,
    text: "Première page",
    start_ms: null,
    end_ms: null,
    page_number: 1,
    chapter_index: null,
    chapter_title: null,
    ...overrides,
  };
}

function finishedJob(): AnalysisJob {
  return {
    id: "job-id",
    source_id: "source-id",
    kind: "analyze_source",
    status: "succeeded",
    stage: "completed",
    progress_current: 2,
    progress_total: 2,
    progress_percent: 100,
    progress_message: "Analyse terminée.",
    error_message: null,
    attempt_count: 1,
    created_at: "2026-09-08T10:00:00Z",
    updated_at: "2026-09-08T10:01:00Z",
    last_activity_at: "2026-09-08T10:01:00Z",
    started_at: "2026-09-08T10:00:00Z",
    finished_at: "2026-09-08T10:01:00Z",
  };
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/sources/source-id"]}>
        <Routes>
          <Route path="/sources/:sourceId" element={<SourceDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(cleanup);

beforeEach(() => {
  vi.clearAllMocks();
  api.getSource.mockResolvedValue(source());
  api.getSourceSegments.mockResolvedValue({
    items: [
      segment(),
      segment({
        id: "segment-1",
        index: 1,
        page_number: 2,
        text: "Deuxième page",
      }),
    ],
    next_cursor: null,
  });
  api.getSourceKnowledgeNodes.mockResolvedValue({ items: [], next_cursor: null });
  api.getSourceAnalysis.mockResolvedValue(finishedJob());
  api.analyzeSource.mockResolvedValue(finishedJob());
});

describe("SourceDetailPage — documents structurés", () => {
  it("affiche les métadonnées et la provenance page d’un PDF", async () => {
    renderPage();

    expect(await screen.findByRole("heading", { name: "Guide pratique" })).toBeVisible();
    expect(screen.getByText("Document PDF")).toBeVisible();
    expect(screen.getByText("Pages extraites")).toBeVisible();
    expect(screen.getByText("2 pages avec provenance conservée.")).toBeVisible();
    expect(screen.getByText("Page 1")).toBeVisible();
    expect(screen.getByText("Page 2")).toBeVisible();
    expect(screen.getByText("fr")).toBeVisible();
  });

  it("affiche les chapitres EPUB dans l’ordre de lecture", async () => {
    api.getSource.mockResolvedValueOnce(
      source({
        type: "epub",
        title: "Roman EPUB",
        original_filename: "roman.epub",
        original_file_path: "originals/source-id/original.epub",
        raw_text: "Préface\n\nLe voyage",
        page_count: null,
        chapter_count: 2,
      }),
    );
    api.getSourceSegments.mockResolvedValueOnce({
      items: [
        segment({
          id: "chapter-0",
          text: "Préface",
          page_number: null,
          chapter_index: 0,
          chapter_title: "Préface",
        }),
        segment({
          id: "chapter-1",
          index: 1,
          text: "Le voyage commence.",
          page_number: null,
          chapter_index: 1,
          chapter_title: "Le voyage",
        }),
      ],
      next_cursor: null,
    });
    renderPage();

    expect(await screen.findByText("Chapitres détectés")).toBeVisible();
    const headings = screen.getAllByText(/Préface|Le voyage/, {
      selector: ".segment-heading strong",
    });
    expect(headings.map((heading) => heading.textContent)).toEqual([
      "Préface",
      "Le voyage",
    ]);
    expect(screen.getByText("Chapitre 1")).toBeVisible();
    expect(screen.getByText("Chapitre 2")).toBeVisible();
  });

  it("envoie un PDF prêt dans le pipeline d’analyse existant", async () => {
    renderPage();
    const analyzeButton = await screen.findByRole("button", {
      name: "Analyser avec l’IA",
    });

    fireEvent.click(analyzeButton);

    await waitFor(() => expect(api.analyzeSource).toHaveBeenCalledWith("source-id"));
  });

  it("bloque l’analyse et explique clairement qu’un PDF nécessite un OCR", async () => {
    api.getSource.mockResolvedValueOnce(
      source({
        title: "Document scanné",
        raw_text: "",
        processing_status: "needs_ocr",
        processing_error:
          "Ce PDF semble nécessiter une reconnaissance OCR, non disponible dans cette version.",
        segment_count: 0,
        page_count: 3,
      }),
    );
    api.getSourceSegments.mockResolvedValueOnce({ items: [], next_cursor: null });
    renderPage();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Ce PDF semble nécessiter une reconnaissance OCR, non disponible dans cette version.",
    );
    const analyzeButton = screen.getByRole("button", {
      name: "Analyse indisponible",
    });
    expect(analyzeButton).toBeDisabled();
    expect(screen.getByText("Aucun texte exploitable n’a été détecté dans ce PDF.")).toBeVisible();
    fireEvent.click(analyzeButton);
    expect(api.analyzeSource).not.toHaveBeenCalled();
  });
});

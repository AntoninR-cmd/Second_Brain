// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SourceDetail } from "../../api/types";
import { AddPage } from "../AddPage";

const api = vi.hoisted(() => ({
  createManualSource: vi.fn(),
  uploadSource: vi.fn(),
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
    title: "Titre extrait",
    author: null,
    original_filename: "livre.pdf",
    processing_status: "ready",
    analysis_status: "not_analyzed",
    created_at: "2026-09-08T10:00:00Z",
    updated_at: "2026-09-08T10:00:00Z",
    raw_text: "Texte extrait",
    original_file_path: "originals/source-id/original.pdf",
    file_sha256: "hash",
    segment_count: 1,
    page_count: 1,
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

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/ajouter"]}>
        <AddPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function openFileForm() {
  fireEvent.click(screen.getByRole("button", { name: /Importer un fichier/i }));
}

afterEach(cleanup);

beforeEach(() => {
  vi.clearAllMocks();
  api.uploadSource.mockResolvedValue(source());
});

describe("AddPage — documents PDF et EPUB", () => {
  it("annonce les quatre formats et configure le sélecteur natif", () => {
    renderPage();
    openFileForm();

    const input = screen.getByLabelText("Fichier à importer");
    expect(input).toHaveAttribute("accept", ".srt,.txt,.pdf,.epub");
    expect(screen.getByText(/\.srt, \.txt, \.pdf et \.epub/i)).toBeVisible();
  });

  it("affiche le titre dérivé d’un PDF mais le laisse au backend tant qu’il n’est pas édité", async () => {
    renderPage();
    openFileForm();
    const file = new File(["%PDF document"], "manuel-pratique.pdf", {
      type: "application/pdf",
    });

    fireEvent.change(screen.getByLabelText("Fichier à importer"), {
      target: { files: [file] },
    });

    expect(screen.getByLabelText(/^Titre/)).toHaveValue("manuel-pratique");
    expect(screen.getByText("PDF")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Importer le fichier" }));

    await waitFor(() => expect(api.uploadSource).toHaveBeenCalledOnce());
    expect(api.uploadSource).toHaveBeenCalledWith({ file });
  });

  it("accepte un EPUB par glisser-déposer et transmet les overrides utilisateur", async () => {
    renderPage();
    openFileForm();
    const file = new File(["epub content"], "roman.epub", {
      type: "application/epub+zip",
    });
    const picker = screen.getByText(/Déposer un fichier ou parcourir/).closest("label");
    expect(picker).not.toBeNull();
    const dataTransfer = {
      dropEffect: "none",
      files: { 0: file, length: 1, item: () => file },
    };

    fireEvent.dragEnter(picker!, { dataTransfer });
    expect(picker).toHaveClass("is-dragging");
    fireEvent.drop(picker!, { dataTransfer });

    expect(picker).not.toHaveClass("is-dragging");
    expect(screen.getByText("EPUB")).toBeVisible();
    fireEvent.change(screen.getByLabelText(/^Titre/), {
      target: { value: "Titre corrigé" },
    });
    fireEvent.change(screen.getByLabelText(/^Auteur/), {
      target: { value: "Autrice corrigée" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Importer le fichier" }));

    await waitFor(() => expect(api.uploadSource).toHaveBeenCalledOnce());
    expect(api.uploadSource).toHaveBeenCalledWith({
      file,
      title: "Titre corrigé",
      author: "Autrice corrigée",
    });
  });

  it("rejette par drop une extension non prise en charge", () => {
    renderPage();
    openFileForm();
    const file = new File(["archive"], "archive.zip", {
      type: "application/zip",
    });
    const picker = screen.getByText(/Déposer un fichier ou parcourir/).closest("label")!;

    fireEvent.drop(picker, {
      dataTransfer: {
        files: { 0: file, length: 1, item: () => file },
      },
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Sélectionnez uniquement un fichier .srt, .txt, .pdf ou .epub.",
    );
    expect(screen.getByRole("button", { name: "Importer le fichier" })).toBeDisabled();
  });

  it("affiche sans l’altérer le diagnostic OCR renvoyé par le backend", async () => {
    api.uploadSource.mockRejectedValueOnce(
      new Error(
        "Ce PDF semble nécessiter une reconnaissance OCR, non disponible dans cette version.",
      ),
    );
    renderPage();
    openFileForm();
    const file = new File(["%PDF image"], "scan.pdf", {
      type: "application/pdf",
    });

    fireEvent.change(screen.getByLabelText("Fichier à importer"), {
      target: { files: [file] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Importer le fichier" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Ce PDF semble nécessiter une reconnaissance OCR, non disponible dans cette version.",
    );
  });
});

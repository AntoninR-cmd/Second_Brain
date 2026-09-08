import { describe, expect, it } from "vitest";

import type { KnowledgeEvidence } from "../api/types";
import {
  getKnowledgeEvidenceLocator,
  getProcessingStatusLabel,
  getSourceTypeLabel,
} from "./sourcePresentation";

function evidence(
  overrides: Partial<KnowledgeEvidence> = {},
): KnowledgeEvidence {
  return {
    id: "evidence",
    passage_id: "passage",
    passage_index: 0,
    original_excerpt: "Extrait",
    start_ms: null,
    end_ms: null,
    first_segment_index: null,
    last_segment_index: null,
    char_start: null,
    char_end: null,
    page_number: null,
    page_end_number: null,
    chapter_index: null,
    chapter_title: null,
    ...overrides,
  };
}

describe("présentation des sources documentaires", () => {
  it("affiche une page PDF et une plage de pages", () => {
    expect(getKnowledgeEvidenceLocator(evidence({ page_number: 84 }))).toBe(
      "Page 84",
    );
    expect(
      getKnowledgeEvidenceLocator(
        evidence({ page_number: 84, page_end_number: 86 }),
      ),
    ).toBe("Pages 84 à 86");
  });

  it("affiche le chapitre EPUB et son titre", () => {
    expect(
      getKnowledgeEvidenceLocator(
        evidence({ chapter_index: 2, chapter_title: "Gestion de la fatigue" }),
      ),
    ).toBe("Chapitre 3 · Gestion de la fatigue");
    expect(
      getKnowledgeEvidenceLocator(
        evidence({ chapter_index: null, chapter_title: "Préface" }),
      ),
    ).toBe("Chapitre · Préface");
  });

  it("conserve la priorité des timestamps SRT", () => {
    expect(
      getKnowledgeEvidenceLocator(
        evidence({ start_ms: 1_000, end_ms: 2_500, page_number: 3 }),
      ),
    ).toBe("00:00:01,000 → 00:00:02,500");
  });

  it("expose les labels PDF, EPUB et OCR", () => {
    expect(getSourceTypeLabel("pdf")).toBe("Document PDF");
    expect(getSourceTypeLabel("epub")).toBe("Livre EPUB");
    expect(getProcessingStatusLabel("needs_ocr")).toBe("OCR nécessaire");
  });
});

import type {
  AnalysisStatus,
  KnowledgeEvidence,
  ProcessingStatus,
  SourceType,
} from "../api/types";

const dateFormatter = new Intl.DateTimeFormat("fr-FR", {
  dateStyle: "medium",
  timeStyle: "short",
});

const SOURCE_TYPE_LABELS: Record<SourceType, string> = {
  manual: "Note manuelle",
  srt: "Sous-titres SRT",
  txt: "Fichier TXT",
};

const PROCESSING_STATUS_LABELS: Record<ProcessingStatus, string> = {
  ready: "Prête",
};

const ANALYSIS_STATUS_LABELS: Record<AnalysisStatus, string> = {
  not_analyzed: "Non analysée",
  queued: "En attente",
  processing: "Analyse en cours",
  analyzed: "Analysée",
  error: "Erreur",
};

export function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Date inconnue"
    : dateFormatter.format(date);
}

export function formatSrtTimestamp(milliseconds: number): string {
  const safeMilliseconds = Math.max(0, Math.trunc(milliseconds));
  const hours = Math.floor(safeMilliseconds / 3_600_000);
  const minutes = Math.floor((safeMilliseconds % 3_600_000) / 60_000);
  const seconds = Math.floor((safeMilliseconds % 60_000) / 1_000);
  const remainder = safeMilliseconds % 1_000;

  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")},${remainder
    .toString()
    .padStart(3, "0")}`;
}

export function getKnowledgeEvidenceLocator(
  evidence: Pick<
    KnowledgeEvidence,
    | "start_ms"
    | "end_ms"
    | "first_segment_index"
    | "last_segment_index"
    | "char_start"
    | "char_end"
    | "passage_index"
  >,
): string {
  if (evidence.start_ms !== null && evidence.end_ms !== null) {
    return `${formatSrtTimestamp(evidence.start_ms)} → ${formatSrtTimestamp(evidence.end_ms)}`;
  }

  if (
    evidence.first_segment_index !== null &&
    evidence.last_segment_index !== null
  ) {
    return evidence.first_segment_index === evidence.last_segment_index
      ? `Segment #${evidence.first_segment_index}`
      : `Segments #${evidence.first_segment_index} à #${evidence.last_segment_index}`;
  }

  if (evidence.char_start !== null && evidence.char_end !== null) {
    return `Caractères ${evidence.char_start} à ${evidence.char_end}`;
  }

  return `Passage #${evidence.passage_index}`;
}

export function getProcessingStatusLabel(status: ProcessingStatus): string {
  return PROCESSING_STATUS_LABELS[status];
}

export function getAnalysisStatusLabel(status: AnalysisStatus): string {
  return ANALYSIS_STATUS_LABELS[status];
}

export function getSourceTypeLabel(type: SourceType): string {
  return SOURCE_TYPE_LABELS[type];
}

export function textExcerpt(value: string, maximumLength = 180): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > maximumLength
    ? `${normalized.slice(0, maximumLength - 3).trimEnd()}…`
    : normalized;
}

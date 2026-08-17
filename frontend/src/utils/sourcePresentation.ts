import type { ProcessingStatus, SourceType } from "../api/types";

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

export function getProcessingStatusLabel(status: ProcessingStatus): string {
  return PROCESSING_STATUS_LABELS[status];
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

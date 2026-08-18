import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useEffect } from "react";
import { Link, useLocation, useParams } from "react-router-dom";

import {
  analyzeSource,
  getReadableError,
  getSource,
  getSourceAnalysis,
  getSourceKnowledgeNodes,
  getSourceSegments,
} from "../api/client";
import type { AnalysisJob } from "../api/types";
import {
  getAnalysisStatusLabel,
  formatDate,
  formatSrtTimestamp,
  getProcessingStatusLabel,
  getSourceTypeLabel,
} from "../utils/sourcePresentation";

interface SourceDetailLocationState {
  flash?: string;
}

const ANALYSIS_STAGE_LABELS: Record<string, string> = {
  queued: "En attente",
  preparing: "Préparation de l’analyse",
  passages: "Préparation des passages",
  preparing_passages: "Préparation des passages",
  extracting: "Analyse des passages",
  analyzing_passages: "Analyse des passages",
  passage_analysis: "Analyse structurée du passage",
  extracting_knowledge: "Extraction des connaissances",
  validating_output: "Validation de la réponse Ollama",
  retrying_passage: "Nouvelle tentative sur le passage",
  summarizing: "Synthèse hiérarchique",
  hierarchical_summary: "Synthèse hiérarchique",
  final_summary: "Résumé final",
  saving: "Enregistrement des résultats",
  resuming: "Reprise de l’analyse",
  recovering: "Récupération du traitement",
  completed: "Analyse terminée",
  failed: "Analyse interrompue",
};

const LLM_CALL_TYPE_LABELS: Record<string, string> = {
  passage_analysis: "Analyse structurée du passage",
  hierarchical_summary: "Synthèse hiérarchique",
  final_summary: "Résumé final",
};

interface AnalysisFailurePresentation {
  category: string;
  detail: string;
  stage: string;
  passageNumber: number | null;
  passageTotal: number | null;
  passageId: string | null;
  attempt: number | null;
  callType: string | null;
  errorCode: string | null;
  errorType: string | null;
}

function getAnalysisStageLabel(stage: string | null): string {
  if (!stage) {
    return "Analyse en cours";
  }

  return ANALYSIS_STAGE_LABELS[stage] ?? humanizeIdentifier(stage);
}

function humanizeIdentifier(value: string): string {
  const normalized = value.replaceAll("_", " ").replaceAll("-", " ").trim();

  return normalized.length > 0
    ? normalized.charAt(0).toLocaleUpperCase("fr-FR") + normalized.slice(1)
    : value;
}

function getAnalysisCallTypeLabel(callType: string): string {
  return LLM_CALL_TYPE_LABELS[callType] ?? humanizeIdentifier(callType);
}

function getAnalysisErrorCategory(
  errorCode: string | null,
  errorType: string | null,
): string {
  const discriminator = `${errorCode ?? ""} ${errorType ?? ""}`.toLowerCase();

  if (
    discriminator.includes("passage_reference") ||
    discriminator.includes("evidence_reference") ||
    discriminator.includes("unknown_passage")
  ) {
    return "Référence de passage invalide";
  }

  if (
    discriminator.includes("validation") ||
    discriminator.includes("pydantic") ||
    discriminator.includes("schema")
  ) {
    return "Validation de la réponse Ollama";
  }

  if (discriminator.includes("json")) {
    return "Réponse JSON Ollama invalide";
  }

  if (discriminator.includes("timeout")) {
    return "Délai d’attente Ollama dépassé";
  }

  if (
    discriminator.includes("model") &&
    (discriminator.includes("missing") ||
      discriminator.includes("absent") ||
      discriminator.includes("not_found"))
  ) {
    return "Modèle Ollama absent";
  }

  if (
    discriminator.includes("unavailable") ||
    discriminator.includes("connection") ||
    discriminator.includes("transport")
  ) {
    return "Ollama indisponible";
  }

  if (
    discriminator.includes("structured_output") ||
    discriminator.includes("invalid_response")
  ) {
    return "Validation de la réponse Ollama";
  }

  if (errorType?.trim()) {
    return humanizeIdentifier(errorType);
  }

  if (errorCode?.trim()) {
    return humanizeIdentifier(errorCode);
  }

  return "Erreur du traitement IA";
}

function getAnalysisFailure(
  job: AnalysisJob | undefined,
  sourceError: string | null,
): AnalysisFailurePresentation | null {
  const isFailed = job
    ? job.status === "failed"
    : Boolean(sourceError?.trim());

  if (!isFailed) {
    return null;
  }

  const errorCode = job?.error_code?.trim() || null;
  const errorType = job?.error_type?.trim() || null;
  const passageIndex =
    job?.error_passage_index ?? job?.failed_passage_index ?? null;
  const passageNumber =
    passageIndex !== null
      ? passageIndex + 1
      : job && job.progress_current > 0
        ? job.progress_current
        : null;
  const passageTotal =
    job && job.progress_total > 0 ? job.progress_total : null;

  return {
    category: getAnalysisErrorCategory(errorCode, errorType),
    detail:
      job?.error_detail?.trim() ||
      job?.error_message?.trim() ||
      sourceError?.trim() ||
      "Le backend n’a fourni aucun détail supplémentaire.",
    stage: getAnalysisStageLabel(job?.error_stage ?? job?.stage ?? null),
    passageNumber,
    passageTotal,
    passageId:
      job?.error_passage_id?.trim() || job?.failed_passage_id?.trim() || null,
    attempt: job?.error_attempt ?? job?.failed_attempt ?? null,
    callType:
      job?.error_call_type?.trim() || job?.failed_call_type?.trim() || null,
    errorCode,
    errorType,
  };
}

function getAnalysisProgressLabel(job: AnalysisJob): string {
  const stageLabel = getAnalysisStageLabel(job.stage);
  const isPassageAnalysis =
    job.stage === "extracting" || job.stage === "analyzing_passages";

  if (isPassageAnalysis && job.progress_total > 0) {
    return `${stageLabel} : ${job.progress_current.toLocaleString("fr-FR")} / ${job.progress_total.toLocaleString("fr-FR")}`;
  }

  return job.progress_message?.trim() || stageLabel;
}

function getSafeProgressPercent(value: number): number {
  return Math.min(100, Math.max(0, Math.round(value)));
}

export function SourceDetailPage() {
  const { sourceId } = useParams();
  const location = useLocation();
  const queryClient = useQueryClient();
  const flash = (location.state as SourceDetailLocationState | null)?.flash;

  const sourceQuery = useQuery({
    queryKey: ["sources", sourceId],
    queryFn: () => getSource(sourceId ?? ""),
    enabled: Boolean(sourceId),
    refetchInterval: (query) => {
      const status = query.state.data?.analysis_status;
      return status === "queued" || status === "processing" ? 1_500 : false;
    },
  });

  const isSrt = sourceQuery.data?.type === "srt";
  const segmentsQuery = useInfiniteQuery({
    queryKey: ["sources", sourceId, "segments"],
    queryFn: ({ pageParam }) =>
      getSourceSegments(sourceId ?? "", pageParam),
    initialPageParam: null as number | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(sourceId) && isSrt,
  });

  const knowledgeQuery = useInfiniteQuery({
    queryKey: ["sources", sourceId, "nodes"],
    queryFn: ({ pageParam }) =>
      getSourceKnowledgeNodes(sourceId ?? "", pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(sourceId),
    refetchInterval:
      sourceQuery.data?.analysis_status === "queued" ||
      sourceQuery.data?.analysis_status === "processing"
        ? 2_000
        : false,
  });

  const analysisMutation = useMutation({
    mutationFn: () => analyzeSource(sourceId ?? ""),
    onSuccess: async (job) => {
      queryClient.setQueryData(["sources", sourceId, "analysis"], job);
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["sources", sourceId],
          exact: true,
        }),
        queryClient.invalidateQueries({
          queryKey: ["sources", sourceId, "nodes"],
        }),
        queryClient.invalidateQueries({
          queryKey: ["sources"],
          exact: true,
        }),
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
      ]);
    },
  });

  const analysisIsActive =
    sourceQuery.data?.analysis_status === "queued" ||
    sourceQuery.data?.analysis_status === "processing";
  const analysisWasStartedHere =
    analysisMutation.data?.source_id === sourceId;
  const analysisHasPersistentJob =
    sourceQuery.data?.analysis_status !== undefined &&
    sourceQuery.data.analysis_status !== "not_analyzed";
  const analysisJobQuery = useQuery({
    queryKey: ["sources", sourceId, "analysis"],
    queryFn: () => getSourceAnalysis(sourceId ?? ""),
    enabled:
      Boolean(sourceId) &&
      (analysisIsActive || analysisWasStartedHere || analysisHasPersistentJob),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "pending" || status === "running" ? 1_000 : false;
    },
  });

  useEffect(() => {
    if (
      sourceQuery.data?.analysis_status === "analyzed" ||
      sourceQuery.data?.analysis_status === "error"
    ) {
      void Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["sources", sourceId, "nodes"],
        }),
        queryClient.invalidateQueries({ queryKey: ["sources"], exact: true }),
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
      ]);
    }
  }, [queryClient, sourceId, sourceQuery.data?.analysis_status]);

  useEffect(() => {
    const status = analysisJobQuery.data?.status;
    if (status === "succeeded" || status === "failed") {
      void Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["sources", sourceId],
          exact: true,
        }),
        queryClient.invalidateQueries({
          queryKey: ["sources", sourceId, "nodes"],
        }),
        queryClient.invalidateQueries({ queryKey: ["sources"], exact: true }),
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
      ]);
    }
  }, [analysisJobQuery.data?.status, queryClient, sourceId]);

  if (!sourceId) {
    return (
      <section className="page narrow-page">
        <p className="eyebrow">Source introuvable</p>
        <h1>Identifiant de source manquant</h1>
        <Link className="button button-primary detail-back-button" to="/sources">
          Revenir aux sources
        </Link>
      </section>
    );
  }

  if (sourceQuery.isPending) {
    return (
      <section className="page">
        <div className="loading-state page-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Chargement de la source…
        </div>
      </section>
    );
  }

  if (sourceQuery.isError) {
    return (
      <section className="page narrow-page">
        <p className="eyebrow">Source indisponible</p>
        <h1>Impossible d’ouvrir cette source</h1>
        <p className="page-introduction">
          {getReadableError(sourceQuery.error)}
        </p>
        <div className="detail-error-actions">
          <Link className="button button-ghost" to="/sources">
            Revenir aux sources
          </Link>
          <button
            className="button button-primary"
            type="button"
            onClick={() => void sourceQuery.refetch()}
          >
            Réessayer
          </button>
        </div>
      </section>
    );
  }

  const source = sourceQuery.data;
  const segments =
    segmentsQuery.data?.pages.flatMap((page) => page.items) ?? [];
  const knowledgeNodes =
    knowledgeQuery.data?.pages.flatMap((page) => page.items) ?? [];
  const analysisJob = analysisJobQuery.data;
  const analysisProgressPercent = analysisJob
    ? getSafeProgressPercent(analysisJob.progress_percent)
    : 0;
  const analysisLastActivityAt = analysisJob
    ? analysisJob.heartbeat_at || analysisJob.last_activity_at
    : null;
  const analysisFailure = getAnalysisFailure(
    analysisJob,
    source.analysis_error,
  );
  const analysisJobIsStale =
    analysisJob?.is_stale === true &&
    (analysisJob.status === "pending" || analysisJob.status === "running");
  const segmentTotal = source.segment_count;
  const analysisIsRunning =
    source.analysis_status === "queued" ||
    source.analysis_status === "processing";
  const analysisButtonDisabled =
    analysisMutation.isPending ||
    analysisIsRunning ||
    source.analysis_status === "analyzed";

  return (
    <section className="page source-detail-page">
      <Link className="back-link" to="/sources">
        <span aria-hidden="true">←</span>
        Toutes les sources
      </Link>

      <header className="page-header source-detail-header">
        <div>
          <div className="detail-labels">
            <span className={`source-type type-${source.type}`}>
              {getSourceTypeLabel(source.type)}
            </span>
            <span className="ready-label">
              {getProcessingStatusLabel(source.processing_status)}
            </span>
            <span
              className={`analysis-status analysis-status-${source.analysis_status}`}
            >
              IA : {getAnalysisStatusLabel(source.analysis_status)}
            </span>
          </div>
          <h1>{source.title}</h1>
          <p className="page-introduction">
            Ajoutée le {formatDate(source.created_at)}
          </p>
        </div>
      </header>

      {flash ? (
        <div className="alert alert-success" role="status">
          <span aria-hidden="true">✓</span>
          <p>{flash}</p>
        </div>
      ) : null}

      <section className="panel detail-panel" aria-labelledby="information-title">
        <div className="panel-header">
          <div>
            <h2 id="information-title">Informations générales</h2>
            <p>Métadonnées conservées avec la source.</p>
          </div>
        </div>
        <dl className="metadata-list">
          <div>
            <dt>Type</dt>
            <dd>{getSourceTypeLabel(source.type)}</dd>
          </div>
          <div>
            <dt>Auteur</dt>
            <dd>{source.author ?? "Non renseigné"}</dd>
          </div>
          <div>
            <dt>Fichier original</dt>
            <dd>{source.original_filename ?? "Sans fichier"}</dd>
          </div>
          <div>
            <dt>Statut</dt>
            <dd>{getProcessingStatusLabel(source.processing_status)}</dd>
          </div>
          {source.type === "srt" ? (
            <div>
              <dt>Segments</dt>
              <dd>{source.segment_count.toLocaleString("fr-FR")}</dd>
            </div>
          ) : null}
          {source.original_file_path ? (
            <div className="metadata-wide">
              <dt>Copie locale</dt>
              <dd>
                <code>{source.original_file_path}</code>
              </dd>
            </div>
          ) : null}
          {source.file_sha256 ? (
            <div className="metadata-wide">
              <dt>Empreinte SHA-256</dt>
              <dd>
                <code>{source.file_sha256}</code>
              </dd>
            </div>
          ) : null}
        </dl>
      </section>

      <section className="panel analysis-panel" aria-labelledby="analysis-title">
        <div className="panel-header analysis-panel-header">
          <div>
            <h2 id="analysis-title">Analyse avec l’IA</h2>
            <p>
              L’analyse est lancée uniquement à votre demande et reste locale.
            </p>
          </div>
          <button
            className="button button-primary"
            type="button"
            disabled={analysisButtonDisabled}
            onClick={() => analysisMutation.mutate()}
          >
            {analysisMutation.isPending ? (
              <>
                <span className="spinner spinner-light" aria-hidden="true" />
                Démarrage…
              </>
            ) : analysisIsRunning ? (
              <>
                <span className="spinner spinner-light" aria-hidden="true" />
                Analyse en cours…
              </>
            ) : source.analysis_status === "analyzed" ? (
              "Analyse terminée"
            ) : source.analysis_status === "error" ? (
              "Réessayer l’analyse"
            ) : (
              "Analyser avec l’IA"
            )}
          </button>
        </div>

        <div className="analysis-state" aria-live="polite">
          <div>
            <span
              className={`analysis-status analysis-status-${source.analysis_status}`}
            >
              {getAnalysisStatusLabel(source.analysis_status)}
            </span>
            <p>
              {source.analysis_status === "not_analyzed"
                ? "Aucune donnée n’est envoyée à Ollama tant que vous ne lancez pas l’analyse."
                : source.analysis_status === "queued"
                  ? "L’analyse est en attente du worker local. Vous pouvez quitter cette page."
                  : source.analysis_status === "processing"
                    ? "Ollama analyse les passages de la source. Cette opération peut prendre plusieurs minutes."
                    : source.analysis_status === "analyzed"
                      ? `${source.knowledge_count.toLocaleString("fr-FR")} connaissance${source.knowledge_count > 1 ? "s" : ""} atomique${source.knowledge_count > 1 ? "s" : ""} enregistrée${source.knowledge_count > 1 ? "s" : ""}.`
                      : "L’analyse n’a pas abouti. La source originale reste intacte."}
            </p>
          </div>
        </div>

        {analysisJob ? (
          <div className="analysis-progress" aria-live="polite">
            <div className="analysis-progress-heading">
              <strong>{getAnalysisProgressLabel(analysisJob)}</strong>
              <span>{analysisProgressPercent} %</span>
            </div>
            <div
              className="analysis-progress-track"
              role="progressbar"
              aria-label="Progression de l’analyse"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={analysisProgressPercent}
            >
              <span
                className="analysis-progress-value"
                style={{ width: `${analysisProgressPercent}%` }}
              />
            </div>
            <div className="analysis-progress-metadata">
              <span>
                Étape : {getAnalysisStageLabel(analysisJob.stage)}
              </span>
              {analysisLastActivityAt ? (
                <time dateTime={analysisLastActivityAt}>
                  Dernière activité : {formatDate(analysisLastActivityAt)}
                </time>
              ) : (
                <span>Dernière activité indisponible</span>
              )}
            </div>
          </div>
        ) : null}

        {analysisJobIsStale ? (
          <div className="alert alert-warning analysis-alert" role="status">
            <span aria-hidden="true">!</span>
            <div>
              <strong>Traitement potentiellement interrompu</strong>
              <p>
                Aucun heartbeat récent n’a été reçu. Après redémarrage du
                backend, l’analyse pourra reprendre au dernier passage validé.
              </p>
            </div>
          </div>
        ) : null}

        {analysisFailure ? (
          <section
            className="analysis-failure"
            role="alert"
            aria-labelledby="analysis-failure-title"
          >
            <div className="analysis-failure-header">
              <span className="analysis-failure-icon" aria-hidden="true">
                !
              </span>
              <div>
                <p>Traitement IA</p>
                <h3 id="analysis-failure-title">Analyse interrompue</h3>
              </div>
            </div>

            <dl className="analysis-failure-details">
              {analysisFailure.passageNumber !== null ? (
                <div>
                  <dt>Passage</dt>
                  <dd>
                    {analysisFailure.passageNumber.toLocaleString("fr-FR")}
                    {analysisFailure.passageTotal !== null
                      ? ` / ${analysisFailure.passageTotal.toLocaleString("fr-FR")}`
                      : ""}
                  </dd>
                </div>
              ) : null}
              <div>
                <dt>Étape</dt>
                <dd>{analysisFailure.stage}</dd>
              </div>
              <div>
                <dt>Erreur</dt>
                <dd>{analysisFailure.category}</dd>
              </div>
              <div className="analysis-failure-detail">
                <dt>Détail</dt>
                <dd>{analysisFailure.detail}</dd>
              </div>
            </dl>

            {analysisFailure.callType ||
            analysisFailure.attempt !== null ||
            analysisFailure.errorCode ||
            analysisFailure.errorType ||
            analysisFailure.passageId ? (
              <details className="analysis-failure-technical">
                <summary>Détails techniques</summary>
                <dl>
                  {analysisFailure.callType ? (
                    <div>
                      <dt>Appel LLM</dt>
                      <dd>
                        {getAnalysisCallTypeLabel(analysisFailure.callType)}
                      </dd>
                    </div>
                  ) : null}
                  {analysisFailure.attempt !== null ? (
                    <div>
                      <dt>Tentative</dt>
                      <dd>{analysisFailure.attempt.toLocaleString("fr-FR")}</dd>
                    </div>
                  ) : null}
                  {analysisFailure.errorType ? (
                    <div>
                      <dt>Type</dt>
                      <dd>
                        <code>{analysisFailure.errorType}</code>
                      </dd>
                    </div>
                  ) : null}
                  {analysisFailure.errorCode ? (
                    <div>
                      <dt>Code</dt>
                      <dd>
                        <code>{analysisFailure.errorCode}</code>
                      </dd>
                    </div>
                  ) : null}
                  {analysisFailure.passageId ? (
                    <div>
                      <dt>Passage ID</dt>
                      <dd>
                        <code>{analysisFailure.passageId}</code>
                      </dd>
                    </div>
                  ) : null}
                </dl>
              </details>
            ) : null}
          </section>
        ) : null}

        {analysisMutation.isError ? (
          <div className="alert alert-error analysis-alert" role="alert">
            <span aria-hidden="true">!</span>
            <p>{getReadableError(analysisMutation.error)}</p>
          </div>
        ) : null}
      </section>

      <section className="panel detail-panel" aria-labelledby="summary-title">
        <div className="panel-header">
          <div>
            <h2 id="summary-title">Résumé détaillé</h2>
            <p>La synthèse enregistrée de cette source.</p>
          </div>
        </div>
        {source.summary ? (
          <div className="summary-content">{source.summary}</div>
        ) : (
          <div className="empty-state compact-empty-state">
            <p>
              {analysisIsRunning
                ? "Le résumé sera affiché à la fin de l’analyse."
                : source.analysis_status === "error"
                  ? "Aucun résumé n’a été enregistré à cause de l’erreur d’analyse."
                  : "Lancez l’analyse IA pour générer et enregistrer un résumé."}
            </p>
          </div>
        )}
      </section>

      <section className="panel detail-panel" aria-labelledby="knowledge-title">
        <div className="panel-header">
          <div>
            <h2 id="knowledge-title">Connaissances atomiques</h2>
            <p>
              Informations autonomes et reliées à leurs passages d’origine.
            </p>
          </div>
          {source.knowledge_count > 0 ? (
            <span className="knowledge-count">
              {source.knowledge_count.toLocaleString("fr-FR")}
            </span>
          ) : null}
        </div>

        {knowledgeQuery.isPending ? (
          <div className="loading-state compact-loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Chargement des connaissances…
          </div>
        ) : knowledgeQuery.isError ? (
          <div className="empty-state error-state compact-knowledge-state" role="alert">
            <h3>Impossible de charger les connaissances</h3>
            <p>{getReadableError(knowledgeQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void knowledgeQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        ) : knowledgeNodes.length === 0 ? (
          <div className="empty-state compact-empty-state">
            <p>
              {analysisIsRunning
                ? "Les connaissances apparaîtront à la fin du traitement."
                : "Aucune connaissance atomique n’est encore associée à cette source."}
            </p>
          </div>
        ) : (
          <>
            <ul className="knowledge-list">
              {knowledgeNodes.map((node) => (
                <li className="knowledge-card" key={node.id}>
                  <h3>
                    <Link to={`/connaissances/${node.id}`}>{node.title}</Link>
                  </h3>
                  <p>{node.content}</p>
                  {node.tags.length > 0 ? (
                    <ul className="tag-list" aria-label={`Tags de ${node.title}`}>
                      {node.tags.map((tag) => (
                        <li key={tag}>#{tag}</li>
                      ))}
                    </ul>
                  ) : null}
                  <Link
                    className="knowledge-open-link"
                    to={`/connaissances/${node.id}`}
                  >
                    Ouvrir la connaissance
                    <span aria-hidden="true">→</span>
                  </Link>
                </li>
              ))}
            </ul>

            {knowledgeQuery.hasNextPage ? (
              <div className="list-actions">
                <button
                  className="button button-secondary"
                  type="button"
                  disabled={knowledgeQuery.isFetchingNextPage}
                  onClick={() => void knowledgeQuery.fetchNextPage()}
                >
                  {knowledgeQuery.isFetchingNextPage ? (
                    <>
                      <span className="spinner" aria-hidden="true" />
                      Chargement…
                    </>
                  ) : (
                    "Afficher plus de connaissances"
                  )}
                </button>
              </div>
            ) : null}
          </>
        )}
      </section>

      <section className="panel detail-panel" aria-labelledby="source-text-title">
        <div className="panel-header">
          <div>
            <h2 id="source-text-title">Texte extrait</h2>
            <p>Contenu textuel conservé dans la base locale.</p>
          </div>
        </div>
        <pre className="source-text">{source.raw_text}</pre>
      </section>

      {source.type === "srt" ? (
        <section className="panel detail-panel" aria-labelledby="segments-title">
          <div className="panel-header">
            <div>
              <h2 id="segments-title">Aperçu des sous-titres</h2>
              <p>
                {segmentTotal.toLocaleString("fr-FR")} segment
                {segmentTotal > 1 ? "s" : ""} avec timestamps précis.
              </p>
            </div>
          </div>

          {segmentsQuery.isPending ? (
            <div className="loading-state" role="status">
              <span className="spinner" aria-hidden="true" />
              Chargement des segments…
            </div>
          ) : segmentsQuery.isError ? (
            <div className="empty-state error-state" role="alert">
              <h3>Impossible de charger les sous-titres</h3>
              <p>{getReadableError(segmentsQuery.error)}</p>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => void segmentsQuery.refetch()}
              >
                Réessayer
              </button>
            </div>
          ) : segments.length === 0 ? (
            <div className="empty-state compact-empty-state">
              <p>Aucun segment n’est associé à cette source.</p>
            </div>
          ) : (
            <>
              <ol className="segment-list">
                {segments.map((segment) => (
                  <li className="segment-card" key={segment.id}>
                    <div className="segment-heading">
                      <strong>#{segment.index}</strong>
                      <time className="segment-time">
                        {segment.start_ms === null || segment.end_ms === null
                          ? "Timestamp indisponible"
                          : `${formatSrtTimestamp(segment.start_ms)} → ${formatSrtTimestamp(segment.end_ms)}`}
                      </time>
                    </div>
                    <p>{segment.text}</p>
                  </li>
                ))}
              </ol>

              {segmentsQuery.hasNextPage ? (
                <div className="list-actions">
                  <button
                    className="button button-secondary"
                    type="button"
                    disabled={segmentsQuery.isFetchingNextPage}
                    onClick={() => void segmentsQuery.fetchNextPage()}
                  >
                    {segmentsQuery.isFetchingNextPage ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        Chargement…
                      </>
                    ) : (
                      "Afficher plus de segments"
                    )}
                  </button>
                </div>
              ) : null}
            </>
          )}
        </section>
      ) : null}
    </section>
  );
}

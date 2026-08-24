import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  getReadableError,
  getSystemReadiness,
  getVectorIndexStatus,
  getVectorJob,
  indexKnowledgeNodes,
  rebuildVectorIndex,
} from "../api/client";
import type { VectorIndexState, VectorJob } from "../api/types";
import { formatDate } from "../utils/sourcePresentation";

const INDEX_STATE_LABELS: Record<VectorIndexState, string> = {
  empty: "Aucune connaissance",
  not_built: "À construire",
  building: "Construction en cours",
  ready: "Prêt",
  stale: "À synchroniser",
  incompatible: "Modèle incompatible",
  unavailable: "Indisponible",
  corrupt: "Endommagé",
};

const PROFILE_STATUS_LABELS = {
  building: "Construction",
  active: "Actif",
  retired: "Retiré",
  failed: "Erreur",
} as const;

function AvailabilityBadge({
  available,
  availableLabel,
  unavailableLabel,
}: {
  available: boolean;
  availableLabel: string;
  unavailableLabel: string;
}) {
  return (
    <span
      className={`availability-badge ${available ? "is-available" : "is-unavailable"}`}
    >
      <span className="availability-dot" aria-hidden="true" />
      {available ? availableLabel : unavailableLabel}
    </span>
  );
}

function getJobMessage(job: VectorJob): string {
  return (
    job.progress_message?.trim() ||
    (job.status === "pending"
      ? "Indexation en attente"
      : job.status === "running"
        ? "Indexation des connaissances"
        : job.status === "succeeded"
          ? "Indexation terminée"
          : "Indexation interrompue")
  );
}

function getJobError(job: VectorJob): string | null {
  return job.error_detail?.trim() || job.error_message?.trim() || null;
}

function getJobLastActivity(job: VectorJob): string | null {
  return job.last_activity_at.trim() || null;
}

function isJobActive(job: VectorJob | null | undefined): boolean {
  return job?.status === "pending" || job?.status === "running";
}

function safeProgress(value: number): number {
  return Math.min(100, Math.max(0, Math.round(value)));
}

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [requestedJobId, setRequestedJobId] = useState<string | null>(null);

  const readinessQuery = useQuery({
    queryKey: ["system", "readiness"],
    queryFn: getSystemReadiness,
    refetchInterval: 30_000,
  });
  const vectorStatusQuery = useQuery({
    queryKey: ["vector-index", "status"],
    queryFn: getVectorIndexStatus,
    refetchInterval: (query) =>
      isJobActive(query.state.data?.active_job) ? 1_500 : 15_000,
  });

  const activeStatusJob = vectorStatusQuery.data?.active_job ?? null;
  const statusJob =
    activeStatusJob ?? vectorStatusQuery.data?.latest_job ?? null;
  const trackedJobId =
    requestedJobId ??
    (isJobActive(activeStatusJob) ? (activeStatusJob?.id ?? null) : null);
  const vectorJobQuery = useQuery({
    queryKey: ["vector-index", "jobs", trackedJobId],
    queryFn: () => getVectorJob(trackedJobId ?? ""),
    enabled: Boolean(trackedJobId),
    refetchInterval: (query) =>
      isJobActive(query.state.data) ? 1_000 : false,
  });

  useEffect(() => {
    const status = vectorJobQuery.data?.status;
    if (status === "succeeded" || status === "failed") {
      void queryClient.invalidateQueries({
        queryKey: ["vector-index", "status"],
      });
    }
  }, [queryClient, vectorJobQuery.data?.status]);

  const indexMutation = useMutation({
    mutationFn: indexKnowledgeNodes,
    onSuccess: (job) => {
      setRequestedJobId(job.id);
      queryClient.setQueryData(["vector-index", "jobs", job.id], job);
      void queryClient.invalidateQueries({
        queryKey: ["vector-index", "status"],
      });
    },
  });
  const rebuildMutation = useMutation({
    mutationFn: rebuildVectorIndex,
    onSuccess: (job) => {
      setRequestedJobId(job.id);
      queryClient.setQueryData(["vector-index", "jobs", job.id], job);
      void queryClient.invalidateQueries({
        queryKey: ["vector-index", "status"],
      });
    },
  });

  function confirmRebuild() {
    const confirmed = window.confirm(
      "Reconstruire entièrement l’index vectoriel ?\n\nLa copie Qdrant sera recréée depuis SQLite. Les connaissances, sources et preuves restent intactes.",
    );
    if (confirmed) {
      rebuildMutation.mutate();
    }
  }

  function refreshAll() {
    void Promise.all([
      readinessQuery.refetch(),
      vectorStatusQuery.refetch(),
      trackedJobId ? vectorJobQuery.refetch() : Promise.resolve(),
    ]);
  }

  const vectorStatus = vectorStatusQuery.data;
  const vectorJob = vectorJobQuery.data ?? statusJob ?? null;
  const vectorJobIsActive = isJobActive(vectorJob);
  const vectorMutationPending =
    indexMutation.isPending || rebuildMutation.isPending;
  const embeddingIsReady =
    vectorStatus?.embedding.ollama_available === true &&
    vectorStatus.embedding.model_available === true;
  const requiresRebuild =
    vectorStatus?.state === "incompatible" ||
    vectorStatus?.state === "corrupt";
  const profileError = vectorStatus?.active_profile?.error_message ?? null;
  const maintenanceItemCount = vectorStatus
    ? vectorStatus.pending_or_stale_nodes +
      vectorStatus.failed_nodes +
      vectorStatus.orphan_points
    : 0;
  const canIndex =
    Boolean(vectorStatus) &&
    embeddingIsReady &&
    !requiresRebuild &&
    !vectorJobIsActive &&
    !vectorMutationPending &&
    maintenanceItemCount > 0;
  const canRebuild =
    Boolean(vectorStatus) &&
    embeddingIsReady &&
    !vectorJobIsActive &&
    !vectorMutationPending &&
    (vectorStatus!.total_nodes > 0 ||
      vectorStatus!.active_profile !== null ||
      vectorStatus!.orphan_points > 0 ||
      requiresRebuild);
  const operationError = indexMutation.error ?? rebuildMutation.error;

  return (
    <section className="page narrow-page settings-page">
      <header className="page-header settings-header">
        <div>
          <p className="eyebrow">Configuration locale</p>
          <h1>Paramètres</h1>
          <p className="page-introduction">
            Vérifiez séparément la génération, les embeddings et l’index
            sémantique. Le navigateur ne contacte jamais Ollama ou Qdrant
            directement.
          </p>
        </div>
        <button
          className="button button-secondary"
          type="button"
          disabled={readinessQuery.isFetching || vectorStatusQuery.isFetching}
          onClick={refreshAll}
        >
          {readinessQuery.isFetching || vectorStatusQuery.isFetching ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Vérification…
            </>
          ) : (
            "Actualiser"
          )}
        </button>
      </header>

      {readinessQuery.isPending ? (
        <section className="panel settings-panel">
          <div className="loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Vérification du modèle de génération…
          </div>
        </section>
      ) : readinessQuery.isError ? (
        <section className="panel settings-panel">
          <div className="empty-state error-state" role="alert">
            <h2>Impossible de vérifier la génération</h2>
            <p>{getReadableError(readinessQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void readinessQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        </section>
      ) : (
        <section className="panel settings-panel" aria-labelledby="generation-title">
          <div className="panel-header">
            <div>
              <h2 id="generation-title">Modèle de génération</h2>
              <p>Utilisé pour les analyses et les réponses RAG sourcées.</p>
            </div>
            <AvailabilityBadge
              available={readinessQuery.data.ollama.available}
              availableLabel="Ollama disponible"
              unavailableLabel="Ollama indisponible"
            />
          </div>

          <dl className="settings-list" aria-live="polite">
            <div>
              <dt>Adresse configurée</dt>
              <dd>
                <code>{readinessQuery.data.ollama.base_url}</code>
              </dd>
            </div>
            <div>
              <dt>Modèle de génération</dt>
              <dd>
                <code>{readinessQuery.data.ollama.configured_model}</code>
              </dd>
            </div>
            <div>
              <dt>Disponibilité du modèle</dt>
              <dd>
                <AvailabilityBadge
                  available={readinessQuery.data.ollama.model_available}
                  availableLabel="Modèle disponible"
                  unavailableLabel="Modèle absent"
                />
              </dd>
            </div>
          </dl>

          {readinessQuery.data.ollama.error ? (
            <div className="settings-error" role="status">
              <strong>Diagnostic</strong>
              <p>{readinessQuery.data.ollama.error}</p>
            </div>
          ) : null}
        </section>
      )}

      {vectorStatusQuery.isPending ? (
        <section className="panel settings-panel">
          <div className="loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Vérification du modèle d’embedding…
          </div>
        </section>
      ) : vectorStatusQuery.isError ? (
        <section className="panel settings-panel">
          <div className="empty-state error-state" role="alert">
            <h2>Impossible de vérifier la mémoire sémantique</h2>
            <p>{getReadableError(vectorStatusQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void vectorStatusQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        </section>
      ) : !vectorStatus ? null : (
        <>
          <section className="panel settings-panel" aria-labelledby="embedding-title">
            <div className="panel-header">
              <div>
                <h2 id="embedding-title">Modèle d’embedding</h2>
                <p>Encode les recherches et le texte titre + contenu.</p>
              </div>
              <AvailabilityBadge
                available={embeddingIsReady}
                availableLabel="Modèle disponible"
                unavailableLabel="Modèle indisponible"
              />
            </div>

            <dl className="settings-list" aria-live="polite">
              <div>
                <dt>Modèle configuré</dt>
                <dd>
                  <code>{vectorStatus.configured_model}</code>
                </dd>
              </div>
              <div>
                <dt>Service Ollama</dt>
                <dd>
                  <AvailabilityBadge
                    available={vectorStatus.embedding.ollama_available}
                    availableLabel="Disponible"
                    unavailableLabel="Indisponible"
                  />
                </dd>
              </div>
              <div>
                <dt>Disponibilité du modèle</dt>
                <dd>
                  <AvailabilityBadge
                    available={vectorStatus.embedding.model_available}
                    availableLabel="Modèle disponible"
                    unavailableLabel="Modèle absent"
                  />
                </dd>
              </div>
            </dl>

            {vectorStatus.embedding.error ? (
              <div className="settings-error" role="status">
                <strong>Diagnostic</strong>
                <p>{vectorStatus.embedding.error}</p>
              </div>
            ) : null}

            <div className="settings-guidance">
              <h3>Aucun téléchargement automatique</h3>
              <p>
                Installez explicitement le modèle configuré avec Ollama avant
                de lancer l’indexation. Les tags et preuves ne sont pas inclus
                dans le texte d’embedding de cette première version.
              </p>
            </div>
          </section>

          <section
            className="panel settings-panel vector-index-panel"
            aria-labelledby="index-title"
          >
            <div className="panel-header vector-index-header">
              <div>
                <h2 id="index-title">Index vectoriel local</h2>
                <p>Copie Qdrant dérivée et reconstructible depuis SQLite.</p>
              </div>
              <span
                className={`availability-badge index-state-${vectorStatus.state} ${
                  vectorStatus.state === "ready"
                    ? "is-available"
                    : vectorStatus.state === "building"
                      ? "is-working"
                      : "is-unavailable"
                }`}
              >
                <span className="availability-dot" aria-hidden="true" />
                {INDEX_STATE_LABELS[vectorStatus.state]}
              </span>
            </div>

            <dl className="index-stat-grid">
              <div>
                <dt>Total SQLite</dt>
                <dd>{vectorStatus.total_nodes.toLocaleString("fr-FR")}</dd>
              </div>
              <div>
                <dt>Indexées</dt>
                <dd>{vectorStatus.indexed_nodes.toLocaleString("fr-FR")}</dd>
              </div>
              <div>
                <dt>À indexer</dt>
                <dd>
                  {vectorStatus.pending_or_stale_nodes.toLocaleString("fr-FR")}
                </dd>
              </div>
              <div>
                <dt>En erreur</dt>
                <dd>{vectorStatus.failed_nodes.toLocaleString("fr-FR")}</dd>
              </div>
              <div>
                <dt>Points orphelins</dt>
                <dd>{vectorStatus.orphan_points.toLocaleString("fr-FR")}</dd>
              </div>
            </dl>

            <dl className="settings-list index-profile-list">
              <div>
                <dt>Modèle de l’index</dt>
                <dd>
                  <code>
                    {vectorStatus.active_profile?.model_name ?? "Aucun profil actif"}
                  </code>
                </dd>
              </div>
              <div>
                <dt>Dimension</dt>
                <dd>
                  {vectorStatus.active_profile
                    ? (vectorStatus.active_profile.dimensions?.toLocaleString(
                        "fr-FR",
                      ) ?? "Non déterminée")
                    : "Non déterminée"}
                </dd>
              </div>
              <div>
                <dt>Distance</dt>
                <dd>{vectorStatus.active_profile?.distance ?? "Cosinus"}</dd>
              </div>
              <div>
                <dt>Génération logique</dt>
                <dd>{vectorStatus.active_profile?.logical_generation ?? "—"}</dd>
              </div>
              <div>
                <dt>État du profil</dt>
                <dd>
                  {vectorStatus.active_profile
                    ? PROFILE_STATUS_LABELS[vectorStatus.active_profile.status]
                    : "Aucun profil"}
                </dd>
              </div>
              <div>
                <dt>Version du texte sémantique</dt>
                <dd>
                  <code>
                    {vectorStatus.active_profile?.semantic_text_version ?? "—"}
                  </code>
                </dd>
              </div>
              <div>
                <dt>Digest du modèle</dt>
                <dd>
                  <code className="vector-profile-digest">
                    {vectorStatus.active_profile?.model_digest ?? "Non fourni"}
                  </code>
                </dd>
              </div>
            </dl>

            {vectorStatus.error || requiresRebuild || profileError ? (
              <div className="settings-error" role="alert">
                <strong>
                  {requiresRebuild
                    ? "Reconstruction nécessaire"
                    : "Diagnostic de l’index"}
                </strong>
                <p>
                  {vectorStatus.error ??
                    profileError ??
                    (vectorStatus.state === "incompatible"
                      ? "L’index existant a été produit par un autre modèle ou une autre dimension."
                      : "L’index dérivé ne peut pas être lu. SQLite reste la source de vérité.")}
                </p>
              </div>
            ) : null}

            {vectorJob ? (
              <div className="analysis-progress index-progress" aria-live="polite">
                <div className="analysis-progress-heading">
                  <strong>{getJobMessage(vectorJob)}</strong>
                  <span>{safeProgress(vectorJob.progress_percent)} %</span>
                </div>
                <div
                  className="analysis-progress-track"
                  role="progressbar"
                  aria-label="Progression de l’indexation"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={safeProgress(vectorJob.progress_percent)}
                >
                  <span
                    className="analysis-progress-value"
                    style={{
                      width: `${safeProgress(vectorJob.progress_percent)}%`,
                    }}
                  />
                </div>
                <div className="analysis-progress-metadata">
                  <span>
                    {vectorJob.progress_current.toLocaleString("fr-FR")} /{" "}
                    {vectorJob.progress_total.toLocaleString("fr-FR")} connaissances
                  </span>
                  {getJobLastActivity(vectorJob) ? (
                    <time dateTime={getJobLastActivity(vectorJob) ?? undefined}>
                      Dernière activité :{" "}
                      {formatDate(getJobLastActivity(vectorJob) ?? "")}
                    </time>
                  ) : null}
                </div>
                {vectorJob.is_stale && vectorJobIsActive ? (
                  <div className="alert alert-warning index-job-warning" role="alert">
                    <div>
                      <strong>Traitement potentiellement interrompu</strong>
                      <p>
                        Le heartbeat n’a pas été mis à jour récemment. Le worker
                        reprendra le job lorsque cela sera possible.
                      </p>
                    </div>
                  </div>
                ) : null}
                {vectorJob.status === "failed" && getJobError(vectorJob) ? (
                  <div className="settings-error" role="alert">
                    <strong>Indexation interrompue</strong>
                    <p>{getJobError(vectorJob)}</p>
                  </div>
                ) : null}
              </div>
            ) : null}

            {operationError ? (
              <div className="settings-error" role="alert">
                <strong>Impossible de démarrer l’opération</strong>
                <p>{getReadableError(operationError)}</p>
              </div>
            ) : null}

            <div className="vector-index-actions">
              <button
                className="button button-primary"
                type="button"
                disabled={!canIndex}
                onClick={() => indexMutation.mutate()}
              >
                {indexMutation.isPending ? (
                  <>
                    <span className="spinner spinner-light" aria-hidden="true" />
                    Démarrage…
                  </>
                ) : (
                  "Indexer les connaissances"
                )}
              </button>
              <button
                className="button button-secondary"
                type="button"
                disabled={!canRebuild}
                onClick={confirmRebuild}
              >
                {rebuildMutation.isPending ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Démarrage…
                  </>
                ) : (
                  "Reconstruire l’index"
                )}
              </button>
            </div>
            <p className="vector-index-safety-note">
              Une panne d’Ollama ou de Qdrant n’efface jamais vos connaissances
              SQLite. Une interruption reprend les éléments déjà indexés.
            </p>
          </section>
        </>
      )}
    </section>
  );
}

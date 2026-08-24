import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  getBrainJob,
  getBrainStatus,
  getReadableError,
  getSystemReadiness,
  getVectorIndexStatus,
  getVectorJob,
  indexKnowledgeNodes,
  rebuildBrain,
  rebuildVectorIndex,
  relabelBrain,
} from "../api/client";
import type {
  BrainJob,
  BrainProfile,
  BrainState,
  VectorIndexState,
  VectorJob,
} from "../api/types";
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

const BRAIN_STATE_LABELS: Record<BrainState, string> = {
  empty: "Aucune connaissance",
  not_built: "À construire",
  building: "Construction en cours",
  ready: "Prêt",
  stale: "À recalculer",
  error: "Erreur",
  vector_index_required: "Index requis",
  unavailable: "Indisponible",
};

const BRAIN_PROFILE_STATUS_LABELS = {
  building: "Construction",
  ready: "Prêt",
  stale: "Obsolète",
  error: "Erreur",
} as const;

const BRAIN_LABEL_STRATEGY_LABELS = {
  deterministic: "Déterministe",
  ollama: "Ollama",
  mixed: "Mixte",
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

function getVectorJobMessage(job: VectorJob): string {
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

function getBrainJobMessage(job: BrainJob): string {
  return (
    job.progress_message?.trim() ||
    (job.status === "pending"
      ? "Construction en attente"
      : job.status === "running"
        ? job.kind === "relabel_brain"
          ? "Génération des labels"
          : "Construction du cerveau"
        : job.status === "succeeded"
          ? job.kind === "relabel_brain"
            ? "Labels régénérés"
            : "Cerveau construit"
          : "Construction interrompue")
  );
}

function getJobError(job: VectorJob | BrainJob): string | null {
  return job.error_detail?.trim() || job.error_message?.trim() || null;
}

function getJobLastActivity(job: VectorJob | BrainJob): string | null {
  return job.last_activity_at.trim() || null;
}

function isJobActive(job: VectorJob | BrainJob | null | undefined): boolean {
  return job?.status === "pending" || job?.status === "running";
}

function safeProgress(value: number): number {
  return Math.min(100, Math.max(0, Math.round(value)));
}

function formatMetric(value: number | null, digits = 3): string {
  return value === null
    ? "—"
    : value.toLocaleString("fr-FR", { maximumFractionDigits: digits });
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${Math.round(milliseconds).toLocaleString("fr-FR")} ms`;
  }
  if (milliseconds < 60_000) {
    return `${(milliseconds / 1_000).toLocaleString("fr-FR", {
      maximumFractionDigits: 2,
    })} s`;
  }
  return `${(milliseconds / 60_000).toLocaleString("fr-FR", {
    maximumFractionDigits: 1,
  })} min`;
}

function brainStateTone(state: BrainState): string {
  if (state === "ready") {
    return "is-available";
  }
  if (
    state === "building" ||
    state === "stale" ||
    state === "vector_index_required"
  ) {
    return "is-working";
  }
  if (state === "error" || state === "unavailable") {
    return "is-unavailable";
  }
  return "is-neutral";
}

function clusterLevelSummary(profile: BrainProfile): string[] {
  return Object.entries(profile.cluster_counts_by_level)
    .sort(([left], [right]) => Number(left) - Number(right))
    .map(
      ([level, count]) =>
        `Niveau ${level} : ${count.toLocaleString("fr-FR")} cluster${count > 1 ? "s" : ""}`,
    );
}

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [requestedJobId, setRequestedJobId] = useState<string | null>(null);
  const [requestedBrainJobId, setRequestedBrainJobId] = useState<string | null>(
    null,
  );

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
  const brainStatusQuery = useQuery({
    queryKey: ["brain", "status"],
    queryFn: getBrainStatus,
    refetchInterval: (query) =>
      isJobActive(query.state.data?.active_job) ? 1_500 : 15_000,
  });
  const activeBrainStatusJob = brainStatusQuery.data?.active_job ?? null;
  const brainStatusJob =
    activeBrainStatusJob ?? brainStatusQuery.data?.latest_job ?? null;
  const trackedBrainJobId =
    requestedBrainJobId ??
    (isJobActive(activeBrainStatusJob)
      ? (activeBrainStatusJob?.id ?? null)
      : null);
  const brainJobQuery = useQuery({
    queryKey: ["brain", "jobs", trackedBrainJobId],
    queryFn: () => getBrainJob(trackedBrainJobId ?? ""),
    enabled: Boolean(trackedBrainJobId),
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

  useEffect(() => {
    const status = brainJobQuery.data?.status;
    if (status === "succeeded" || status === "failed") {
      void queryClient.invalidateQueries({ queryKey: ["brain", "status"] });
    }
  }, [brainJobQuery.data?.status, queryClient]);

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
  const rebuildBrainMutation = useMutation({
    mutationFn: rebuildBrain,
    onSuccess: (job) => {
      setRequestedBrainJobId(job.id);
      queryClient.setQueryData(["brain", "jobs", job.id], job);
      void queryClient.invalidateQueries({ queryKey: ["brain", "status"] });
    },
  });
  const relabelBrainMutation = useMutation({
    mutationFn: relabelBrain,
    onSuccess: (job) => {
      setRequestedBrainJobId(job.id);
      queryClient.setQueryData(["brain", "jobs", job.id], job);
      void queryClient.invalidateQueries({ queryKey: ["brain", "status"] });
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

  function confirmBrainRebuild() {
    const confirmed = window.confirm(
      "Recalculer le modèle mathématique du cerveau ?\n\nLes relations, clusters, labels et coordonnées seront reconstruits à partir de l’index actif. Les sources, connaissances, preuves, Qdrant et le RAG restent intacts. L’ancienne version reste disponible jusqu’à la fin du calcul.",
    );
    if (confirmed) {
      rebuildBrainMutation.mutate();
    }
  }

  function confirmBrainRelabel() {
    const confirmed = window.confirm(
      "Régénérer les labels des clusters ?\n\nLa structure mathématique et les coordonnées ne seront pas recalculées. Si Ollama est indisponible, les labels actuels restent utilisables.",
    );
    if (confirmed) {
      relabelBrainMutation.mutate();
    }
  }

  function refreshAll() {
    void Promise.all([
      readinessQuery.refetch(),
      vectorStatusQuery.refetch(),
      brainStatusQuery.refetch(),
      trackedJobId ? vectorJobQuery.refetch() : Promise.resolve(),
      trackedBrainJobId ? brainJobQuery.refetch() : Promise.resolve(),
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
  const brainStatus = brainStatusQuery.data;
  const brainJob = brainJobQuery.data ?? brainStatusJob ?? null;
  const brainJobIsActive = isJobActive(brainJob);
  const brainMutationPending =
    rebuildBrainMutation.isPending || relabelBrainMutation.isPending;
  const displayedBrainProfile =
    brainStatus?.active_profile ?? brainStatus?.building_profile ?? null;
  const canRebuildBrain =
    brainStatus?.can_rebuild === true &&
    !brainJobIsActive &&
    !brainMutationPending;
  const canRelabelBrain =
    brainStatus?.can_relabel === true &&
    !brainJobIsActive &&
    !brainMutationPending;
  const brainOperationError =
    rebuildBrainMutation.error ?? relabelBrainMutation.error;
  const brainClusterLevels = displayedBrainProfile
    ? clusterLevelSummary(displayedBrainProfile)
    : [];
  const settingsAreRefreshing =
    readinessQuery.isFetching ||
    vectorStatusQuery.isFetching ||
    brainStatusQuery.isFetching;

  return (
    <section className="page narrow-page settings-page">
      <header className="page-header settings-header">
        <div>
          <p className="eyebrow">Configuration locale</p>
          <h1>Paramètres</h1>
          <p className="page-introduction">
            Vérifiez séparément la génération, les embeddings et l’index
            sémantique, puis construisez le modèle mathématique du cerveau. Le
            navigateur ne contacte jamais Ollama ou Qdrant directement.
          </p>
        </div>
        <button
          className="button button-secondary"
          type="button"
          disabled={settingsAreRefreshing}
          onClick={refreshAll}
        >
          {settingsAreRefreshing ? (
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
                  <strong>{getVectorJobMessage(vectorJob)}</strong>
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

      {brainStatusQuery.isPending ? (
        <section className="panel settings-panel">
          <div className="loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Vérification du modèle mathématique du cerveau…
          </div>
        </section>
      ) : brainStatusQuery.isError ? (
        <section className="panel settings-panel">
          <div className="empty-state error-state" role="alert">
            <h2>Impossible de vérifier le cerveau</h2>
            <p>{getReadableError(brainStatusQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void brainStatusQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        </section>
      ) : !brainStatus ? null : (
        <section
          className="panel settings-panel brain-model-panel"
          aria-labelledby="brain-model-title"
        >
          <div className="panel-header brain-model-header">
            <div>
              <h2 id="brain-model-title">Modèle mathématique du cerveau</h2>
              <p>
                Représentation dérivée des embeddings : relations, clusters,
                hiérarchie, labels et coordonnées 2D.
              </p>
            </div>
            <span
              className={`availability-badge brain-state-${brainStatus.state} ${brainStateTone(brainStatus.state)}`}
            >
              <span className="availability-dot" aria-hidden="true" />
              {BRAIN_STATE_LABELS[brainStatus.state]}
            </span>
          </div>

          {displayedBrainProfile ? (
            <>
              <dl className="brain-stat-grid">
                <div>
                  <dt>Connaissances</dt>
                  <dd>
                    {displayedBrainProfile.knowledge_node_count.toLocaleString(
                      "fr-FR",
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Relations</dt>
                  <dd>
                    {displayedBrainProfile.edge_count.toLocaleString("fr-FR")}
                  </dd>
                </div>
                <div>
                  <dt>Clusters</dt>
                  <dd>
                    {displayedBrainProfile.cluster_count.toLocaleString("fr-FR")}
                  </dd>
                </div>
                <div>
                  <dt>Non assignées</dt>
                  <dd>
                    {displayedBrainProfile.unassigned_node_count.toLocaleString(
                      "fr-FR",
                    )}
                  </dd>
                </div>
              </dl>

              <dl className="settings-list brain-profile-list">
                <div>
                  <dt>Génération du cerveau</dt>
                  <dd>{displayedBrainProfile.logical_generation}</dd>
                </div>
                <div>
                  <dt>État du profil</dt>
                  <dd>
                    {BRAIN_PROFILE_STATUS_LABELS[displayedBrainProfile.status]}
                  </dd>
                </div>
                <div>
                  <dt>Modèle d’embedding</dt>
                  <dd>
                    <code>{displayedBrainProfile.embedding_model_name}</code>
                  </dd>
                </div>
                <div>
                  <dt>Dimension</dt>
                  <dd>
                    {displayedBrainProfile.embedding_dimensions.toLocaleString(
                      "fr-FR",
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Génération de l’index</dt>
                  <dd>{displayedBrainProfile.embedding_logical_generation}</dd>
                </div>
                <div>
                  <dt>Algorithmes</dt>
                  <dd>
                    <code>{displayedBrainProfile.algorithm_version}</code>
                  </dd>
                </div>
                <div>
                  <dt>Stratégie de labels</dt>
                  <dd>
                    {BRAIN_LABEL_STRATEGY_LABELS[
                      displayedBrainProfile.label_strategy
                    ]}
                    {displayedBrainProfile.label_model_name
                      ? ` · ${displayedBrainProfile.label_model_name}`
                      : ""}
                  </dd>
                </div>
                <div>
                  <dt>Profil activé</dt>
                  <dd>
                    {displayedBrainProfile.activated_at
                      ? formatDate(displayedBrainProfile.activated_at)
                      : displayedBrainProfile.completed_at
                        ? formatDate(displayedBrainProfile.completed_at)
                        : "Construction en cours"}
                  </dd>
                </div>
                <div>
                  <dt>Digest du modèle</dt>
                  <dd>
                    <code className="vector-profile-digest">
                      {displayedBrainProfile.embedding_model_digest ?? "Non fourni"}
                    </code>
                  </dd>
                </div>
              </dl>

              <div className="brain-metrics-section">
                <div>
                  <h3>Hiérarchie</h3>
                  {brainClusterLevels.length > 0 ? (
                    <ul className="brain-level-list">
                      {brainClusterLevels.map((summary) => (
                        <li key={summary}>{summary}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>Aucun niveau de cluster pour ce corpus.</p>
                  )}
                </div>
                <div>
                  <h3>Similarités des relations</h3>
                  <dl className="brain-inline-stats">
                    <div>
                      <dt>Min.</dt>
                      <dd>{formatMetric(displayedBrainProfile.similarity.minimum)}</dd>
                    </div>
                    <div>
                      <dt>Moy.</dt>
                      <dd>{formatMetric(displayedBrainProfile.similarity.mean)}</dd>
                    </div>
                    <div>
                      <dt>Médiane</dt>
                      <dd>{formatMetric(displayedBrainProfile.similarity.median)}</dd>
                    </div>
                    <div>
                      <dt>Max.</dt>
                      <dd>{formatMetric(displayedBrainProfile.similarity.maximum)}</dd>
                    </div>
                  </dl>
                </div>
                <div>
                  <h3>Taille des clusters</h3>
                  <dl className="brain-inline-stats">
                    <div>
                      <dt>Min.</dt>
                      <dd>{formatMetric(displayedBrainProfile.cluster_sizes.minimum, 1)}</dd>
                    </div>
                    <div>
                      <dt>Moy.</dt>
                      <dd>{formatMetric(displayedBrainProfile.cluster_sizes.mean, 1)}</dd>
                    </div>
                    <div>
                      <dt>Max.</dt>
                      <dd>{formatMetric(displayedBrainProfile.cluster_sizes.maximum, 1)}</dd>
                    </div>
                  </dl>
                </div>
              </div>

              <dl className="brain-duration-grid">
                <div>
                  <dt>Relations</dt>
                  <dd>{formatDuration(displayedBrainProfile.relations_duration_ms)}</dd>
                </div>
                <div>
                  <dt>Clustering</dt>
                  <dd>{formatDuration(displayedBrainProfile.clustering_duration_ms)}</dd>
                </div>
                <div>
                  <dt>UMAP</dt>
                  <dd>{formatDuration(displayedBrainProfile.umap_duration_ms)}</dd>
                </div>
                <div>
                  <dt>Labels</dt>
                  <dd>{formatDuration(displayedBrainProfile.labeling_duration_ms)}</dd>
                </div>
                <div>
                  <dt>Total</dt>
                  <dd>{formatDuration(displayedBrainProfile.total_duration_ms)}</dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="brain-empty-state">
              <strong>Aucun modèle mathématique construit</strong>
              <p>
                Indexez d’abord les connaissances, puis lancez une construction.
                Aucun calcul n’est exécuté à l’ouverture de cette page.
              </p>
            </div>
          )}

          {brainStatus.stale_reasons.length > 0 ? (
            <div className="alert alert-warning brain-stale-warning" role="alert">
              <div>
                <strong>Recalcul recommandé</strong>
                <ul>
                  {brainStatus.stale_reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}

          {brainStatus.error ? (
            <div className="settings-error" role="alert">
              <strong>Diagnostic du cerveau</strong>
              <p>{brainStatus.error}</p>
            </div>
          ) : null}

          {displayedBrainProfile?.error_message &&
          displayedBrainProfile.error_message !== brainStatus.error ? (
            <div className="settings-error" role="alert">
              <strong>Diagnostic du cerveau</strong>
              <p>{displayedBrainProfile.error_message}</p>
            </div>
          ) : null}

          {brainJob ? (
            <div className="analysis-progress brain-progress" aria-live="polite">
              <div className="analysis-progress-heading">
                <strong>{getBrainJobMessage(brainJob)}</strong>
                <span>{safeProgress(brainJob.progress_percent)} %</span>
              </div>
              <div
                className="analysis-progress-track"
                role="progressbar"
                aria-label="Progression de la construction du cerveau"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={safeProgress(brainJob.progress_percent)}
              >
                <span
                  className="analysis-progress-value"
                  style={{ width: `${safeProgress(brainJob.progress_percent)}%` }}
                />
              </div>
              <div className="analysis-progress-metadata">
                <span>
                  {brainJob.progress_total > 0
                    ? `${brainJob.progress_current.toLocaleString("fr-FR")} / ${brainJob.progress_total.toLocaleString("fr-FR")}`
                    : (brainJob.stage ?? "Préparation")}
                </span>
                {getJobLastActivity(brainJob) ? (
                  <time dateTime={getJobLastActivity(brainJob) ?? undefined}>
                    Dernière activité :{" "}
                    {formatDate(getJobLastActivity(brainJob) ?? "")}
                  </time>
                ) : null}
              </div>
              {brainJob.is_stale && brainJobIsActive ? (
                <div className="alert alert-warning index-job-warning" role="alert">
                  <div>
                    <strong>Traitement potentiellement interrompu</strong>
                    <p>
                      Le heartbeat n’a pas été mis à jour récemment. Le worker
                      reprendra le calcul sans altérer l’ancien profil actif.
                    </p>
                  </div>
                </div>
              ) : null}
              {brainJob.status === "failed" && getJobError(brainJob) ? (
                <div className="settings-error" role="alert">
                  <strong>Construction interrompue</strong>
                  <p>{getJobError(brainJob)}</p>
                </div>
              ) : null}
            </div>
          ) : null}

          {brainOperationError ? (
            <div className="settings-error" role="alert">
              <strong>Impossible de démarrer l’opération</strong>
              <p>{getReadableError(brainOperationError)}</p>
            </div>
          ) : null}

          <div className="brain-model-actions">
            <button
              className="button button-primary"
              type="button"
              disabled={!canRebuildBrain}
              onClick={confirmBrainRebuild}
            >
              {rebuildBrainMutation.isPending ? (
                <>
                  <span className="spinner spinner-light" aria-hidden="true" />
                  Démarrage…
                </>
              ) : brainStatus.active_profile ? (
                "Recalculer le cerveau"
              ) : (
                "Construire le cerveau"
              )}
            </button>
            <button
              className="button button-secondary"
              type="button"
              disabled={!canRelabelBrain}
              onClick={confirmBrainRelabel}
            >
              {relabelBrainMutation.isPending ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Démarrage…
                </>
              ) : (
                "Régénérer les labels"
              )}
            </button>
          </div>
          <p className="brain-model-safety-note">
            Ce modèle est entièrement dérivé et reconstructible. Une erreur de
            calcul ne modifie jamais les sources, KnowledgeNodes, preuves,
            embeddings, Qdrant ou réponses RAG.
          </p>
        </section>
      )}
    </section>
  );
}

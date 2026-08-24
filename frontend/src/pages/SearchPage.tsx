import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  answerWithRag,
  getReadableError,
  getSystemReadiness,
  getVectorIndexStatus,
} from "../api/client";
import type {
  KnowledgeEvidence,
  RagAnswerResponse,
  RagKnowledge,
  RagMode,
  RagTimings,
  VectorIndexState,
} from "../api/types";
import {
  formatSrtTimestamp,
  getSourceTypeLabel,
} from "../utils/sourcePresentation";

const INDEX_STATE_LABELS: Record<VectorIndexState, string> = {
  empty: "Aucune connaissance",
  not_built: "Index à construire",
  building: "Indexation en cours",
  ready: "Index prêt",
  stale: "Index incomplet",
  incompatible: "Index incompatible",
  unavailable: "Index indisponible",
  corrupt: "Index endommagé",
};

const scoreFormatter = new Intl.NumberFormat("fr-FR", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

const durationFormatter = new Intl.NumberFormat("fr-FR", {
  maximumFractionDigits: 0,
});

const RAG_MODE_OPTIONS: ReadonlyArray<{
  value: RagMode;
  label: string;
  description: string;
}> = [
  {
    value: "brain_only",
    label: "Second cerveau uniquement",
    description: "Répond exclusivement à partir de vos connaissances indexées.",
  },
  {
    value: "brain_plus_model",
    label: "Second cerveau + modèle",
    description:
      "Sépare les informations de votre mémoire des compléments généraux du modèle.",
  },
];

function parseRagMode(value: string | null): RagMode {
  return value === "brain_plus_model" ? value : "brain_only";
}

function ragCacheKey(question: string, mode: RagMode) {
  return ["rag-answer", question, mode] as const;
}

function formatDuration(durationMs: number): string {
  if (durationMs >= 1_000) {
    return `${(durationMs / 1_000).toLocaleString("fr-FR", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 2,
    })} s`;
  }

  return `${durationFormatter.format(durationMs)} ms`;
}

function getEvidenceLocator(evidence: KnowledgeEvidence): string {
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

function canSearchIndex(
  state: VectorIndexState,
  indexedNodes: number,
  embeddingReady: boolean,
): boolean {
  return (
    embeddingReady &&
    indexedNodes > 0 &&
    (state === "ready" || state === "stale")
  );
}

function IndexNotice({
  state,
  indexedNodes,
  totalNodes,
  embeddingReady,
  embeddingError,
}: {
  state: VectorIndexState;
  indexedNodes: number;
  totalNodes: number;
  embeddingReady: boolean;
  embeddingError: string | null;
}) {
  if (state === "ready" && embeddingReady) {
    return null;
  }

  const embeddingUnavailable =
    !embeddingReady &&
    indexedNodes > 0 &&
    (state === "ready" || state === "stale");
  const isBlocking = !canSearchIndex(state, indexedNodes, embeddingReady);
  let label = INDEX_STATE_LABELS[state];
  let message = "L’index sémantique doit être préparé avant la recherche.";

  if (embeddingUnavailable) {
    label = "Modèle d’embedding indisponible";
    message =
      embeddingError ??
      "Ollama ou le modèle d’embedding configuré est indisponible. La recherche nécessite ce modèle pour encoder votre requête.";
  } else if (state === "empty") {
    message =
      totalNodes === 0
        ? "Aucune connaissance atomique n’est encore disponible à indexer."
        : "L’index ne contient encore aucune connaissance.";
  } else if (state === "building") {
    message =
      "L’indexation est en cours. La recherche sera disponible après validation complète de l’index.";
  } else if (state === "stale") {
    message =
      "Certaines connaissances sont nouvelles ou modifiées. Les résultats disponibles peuvent être incomplets.";
  } else if (state === "incompatible") {
    message =
      "Le modèle configuré ne correspond pas à l’index existant. Une reconstruction est nécessaire.";
  } else if (state === "unavailable") {
    message =
      "L’index vectoriel local est indisponible. Vos connaissances SQLite restent intactes.";
  } else if (state === "corrupt") {
    message =
      "L’index vectoriel ne peut pas être lu. Reconstruisez sa copie dérivée depuis les paramètres.";
  }

  return (
    <div
      className={`search-index-notice ${isBlocking ? "is-blocking" : ""}`}
      role={isBlocking ? "alert" : "status"}
    >
      <div>
        <strong>{label}</strong>
        <p>{message}</p>
      </div>
      <Link className="button button-secondary" to="/parametres">
        Ouvrir les paramètres
      </Link>
    </div>
  );
}

function GenerationNotice({
  ollamaAvailable,
  modelAvailable,
  model,
  error,
}: {
  ollamaAvailable: boolean;
  modelAvailable: boolean;
  model: string;
  error: string | null;
}) {
  if (ollamaAvailable && modelAvailable) {
    return null;
  }

  return (
    <div className="search-index-notice is-blocking" role="alert">
      <div>
        <strong>Modèle de génération indisponible</strong>
        <p>
          {error ??
            (ollamaAvailable
              ? `Le modèle ${model} n’est pas installé dans Ollama.`
              : "Ollama ne répond pas. Une réponse sourcée ne peut pas être générée pour le moment.")}
        </p>
      </div>
      <Link className="button button-secondary" to="/parametres">
        Ouvrir les paramètres
      </Link>
    </div>
  );
}

function CitedAnswer({
  text,
  usedKnowledge,
  returnPath,
}: {
  text: string;
  usedKnowledge: RagKnowledge[];
  returnPath: string;
}) {
  const references = new Map(
    usedKnowledge
      .filter(
        (item): item is RagKnowledge & { context_id: string } =>
          item.context_id !== null,
      )
      .map((item) => [item.context_id, item]),
  );

  return (
    <p className="rag-answer-text">
      {text.split(/(\[K[1-9]\d*\])/g).map((part, index) => {
        const contextId = /^\[(K[1-9]\d*)\]$/.exec(part)?.[1];
        const knowledge = contextId ? references.get(contextId) : undefined;

        return knowledge ? (
          <Link
            className="rag-inline-citation"
            key={`${part}-${index}`}
            to={knowledge.href}
            state={{ fromSearch: returnPath }}
            aria-label={`${part}, ouvrir ${knowledge.knowledge_node.title}`}
          >
            {part}
          </Link>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        );
      })}
    </p>
  );
}

function RagKnowledgeCard({
  item,
  rank,
  returnPath,
}: {
  item: RagKnowledge;
  rank: number;
  returnPath: string;
}) {
  const node = item.knowledge_node;
  const primaryEvidence = item.evidences[0];

  return (
    <li className="semantic-result-card">
      <div className="semantic-result-rank" aria-hidden="true">
        {rank}
      </div>
      <article>
        <div className="semantic-result-heading">
          <div>
            <div className="rag-knowledge-labels">
              {item.context_id ? <strong>[{item.context_id}]</strong> : null}
              {item.used ? <span className="rag-status is-used">Citée</span> : null}
              {!item.used && item.provided_to_model ? (
                <span className="rag-status">Fournie au modèle</span>
              ) : null}
              {!item.provided_to_model ? (
                <span className="rag-status is-muted">Hors contexte final</span>
              ) : null}
            </div>
            <h3>
              <Link
                to={item.href}
                state={{ fromSearch: returnPath }}
              >
                {node.title}
              </Link>
            </h3>
          </div>
          <span className="similarity-score">
            similarité : {scoreFormatter.format(item.score)}
          </span>
        </div>
        <p className="semantic-result-content">{node.content}</p>

        {node.tags.length > 0 ? (
          <ul className="tag-list" aria-label={`Tags de ${node.title}`}>
            {node.tags.map((tag) => (
              <li key={tag}>#{tag}</li>
            ))}
          </ul>
        ) : null}

        <div className="semantic-result-source">
          <span>{getSourceTypeLabel(item.source.type)}</span>
          <Link to={`/sources/${item.source.id}`}>{item.source.title}</Link>
          {item.source.author ? <span>par {item.source.author}</span> : null}
        </div>

        {primaryEvidence ? (
          <div className="semantic-result-evidence">
            <strong>{getEvidenceLocator(primaryEvidence)}</strong>
            <blockquote>{primaryEvidence.original_excerpt}</blockquote>
          </div>
        ) : null}

        <Link
          className="knowledge-open-link"
          to={item.href}
          state={{ fromSearch: returnPath }}
        >
          Ouvrir la connaissance
          <span aria-hidden="true">→</span>
        </Link>
      </article>
    </li>
  );
}

function RagTimingsDetails({ timings }: { timings: RagTimings }) {
  return (
    <details className="rag-timings">
      <summary>
        Détails de performance · total {formatDuration(timings.total_ms)}
      </summary>
      <dl>
        <div>
          <dt>Disponibilité</dt>
          <dd>{formatDuration(timings.readiness_ms)}</dd>
        </div>
        <div>
          <dt>Embedding de la question</dt>
          <dd>{formatDuration(timings.embedding_ms)}</dd>
        </div>
        <div>
          <dt>Recherche Qdrant</dt>
          <dd>{formatDuration(timings.qdrant_ms)}</dd>
        </div>
        <div>
          <dt>Chargement SQLite</dt>
          <dd>{formatDuration(timings.retrieval_sqlite_ms)}</dd>
        </div>
        <div>
          <dt>Construction du contexte</dt>
          <dd>{formatDuration(timings.context_build_ms)}</dd>
        </div>
        <div>
          <dt>Génération Ollama</dt>
          <dd>{formatDuration(timings.generation_ms)}</dd>
        </div>
        <div>
          <dt>Validation des sources</dt>
          <dd>{formatDuration(timings.provenance_validation_ms)}</dd>
        </div>
        {timings.prompt_eval_count !== null ? (
          <div>
            <dt>Tokens d’entrée</dt>
            <dd>{timings.prompt_eval_count.toLocaleString("fr-FR")}</dd>
          </div>
        ) : null}
        {timings.eval_count !== null ? (
          <div>
            <dt>Tokens générés</dt>
            <dd>{timings.eval_count.toLocaleString("fr-FR")}</dd>
          </div>
        ) : null}
      </dl>
    </details>
  );
}

export function SearchPage() {
  const queryClient = useQueryClient();
  const [searchParameters, setSearchParameters] = useSearchParams();
  const submittedQuestion = searchParameters.get("q")?.trim() ?? "";
  const submittedMode = parseRagMode(searchParameters.get("mode"));
  const [draftQuestion, setDraftQuestion] = useState(submittedQuestion);
  const [draftMode, setDraftMode] = useState<RagMode>(submittedMode);
  const [response, setResponse] = useState<RagAnswerResponse | null>(() =>
    submittedQuestion
      ? (queryClient.getQueryData<RagAnswerResponse>(
          ragCacheKey(submittedQuestion, submittedMode),
        ) ?? null)
      : null,
  );

  useEffect(() => {
    setDraftQuestion(submittedQuestion);
    setDraftMode(submittedMode);
    setResponse(
      submittedQuestion
        ? (queryClient.getQueryData<RagAnswerResponse>(
            ragCacheKey(submittedQuestion, submittedMode),
          ) ?? null)
        : null,
    );
  }, [queryClient, submittedMode, submittedQuestion]);

  const indexQuery = useQuery({
    queryKey: ["vector-index", "status"],
    queryFn: getVectorIndexStatus,
    refetchInterval: (query) =>
      query.state.data?.state === "building" ? 1_500 : 15_000,
  });
  const readinessQuery = useQuery({
    queryKey: ["system", "readiness"],
    queryFn: getSystemReadiness,
    refetchInterval: 30_000,
  });

  const embeddingIsReady = Boolean(
    indexQuery.data?.embedding.ollama_available &&
      indexQuery.data.embedding.model_available,
  );
  const indexCanBeSearched = indexQuery.data
    ? canSearchIndex(
        indexQuery.data.state,
        indexQuery.data.indexed_nodes,
        embeddingIsReady,
      )
    : false;
  const generationIsReady = Boolean(
    readinessQuery.data?.ollama.available &&
      readinessQuery.data.ollama.model_available,
  );
  const ragCanRun = indexCanBeSearched && generationIsReady;
  const ragMutation = useMutation({
    mutationFn: answerWithRag,
    retry: 0,
    onSuccess: (answer) => {
      queryClient.setQueryData(
        ragCacheKey(answer.question, answer.mode),
        answer,
      );
      setResponse(answer);
    },
  });

  function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuestion = draftQuestion.trim().replace(/\s+/g, " ");

    if (!normalizedQuestion) {
      setSearchParameters({});
      setResponse(null);
      return;
    }

    if (!ragCanRun || normalizedQuestion.length < 2) {
      return;
    }

    const cacheKey = ragCacheKey(normalizedQuestion, draftMode);
    queryClient.removeQueries({ queryKey: cacheKey, exact: true });
    setResponse(null);
    setSearchParameters({ q: normalizedQuestion, mode: draftMode });
    ragMutation.mutate({ question: normalizedQuestion, mode: draftMode });
  }

  const responseSearchParameters = new URLSearchParams();
  if (response) {
    responseSearchParameters.set("q", response.question);
    responseSearchParameters.set("mode", response.mode);
  }
  const returnPath = response
    ? `/recherche?${responseSearchParameters.toString()}`
    : "/recherche";
  const providedKnowledgeCount =
    response?.retrieved_knowledge.filter((item) => item.provided_to_model)
      .length ?? 0;

  return (
    <section className="page search-page">
      <header className="page-header search-header">
        <div>
          <p className="eyebrow">RAG local et sourcé</p>
          <h1>Recherche dans mon cerveau</h1>
          <p className="page-introduction">
            Posez une question : les connaissances sont retrouvées par leur
            sens, puis Ollama construit une réponse dont chaque source reste
            vérifiable.
          </p>
        </div>
      </header>

      <form
        className="panel semantic-search-form rag-question-form"
        role="search"
        onSubmit={submitQuestion}
      >
        <label htmlFor="semantic-query">Votre question</label>
        <div className="semantic-search-controls">
          <input
            id="semantic-query"
            name="q"
            type="search"
            value={draftQuestion}
            placeholder="Comment empêcher une peinture de se décoller d’un pare-chocs ?"
            autoComplete="off"
            aria-describedby="semantic-search-help"
            onChange={(event) => setDraftQuestion(event.target.value)}
          />
          <button
            className="button button-primary"
            type="submit"
            disabled={
              draftQuestion.trim().length < 2 ||
              indexQuery.isPending ||
              readinessQuery.isPending ||
              !ragCanRun ||
              ragMutation.isPending
            }
          >
            {ragMutation.isPending ? (
              <>
                <span className="spinner spinner-light" aria-hidden="true" />
                Génération…
              </>
            ) : (
              "Envoyer"
            )}
          </button>
        </div>
        <p id="semantic-search-help">
          La recherche, le contexte et la génération restent sur cet ordinateur.
        </p>

        <fieldset className="rag-mode-fieldset">
          <legend>Mode de réponse</legend>
          <div className="rag-mode-options">
            {RAG_MODE_OPTIONS.map((option) => (
              <label
                className={`rag-mode-option${draftMode === option.value ? " is-selected" : ""}`}
                key={option.value}
              >
                <input
                  type="radio"
                  name="rag-mode"
                  value={option.value}
                  checked={draftMode === option.value}
                  onChange={() => setDraftMode(option.value)}
                />
                <span>
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      </form>

      {indexQuery.isPending ? (
        <div className="loading-state search-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Vérification de l’index sémantique…
        </div>
      ) : indexQuery.isError ? (
        <div className="empty-state error-state search-state" role="alert">
          <h2>Impossible de vérifier l’index</h2>
          <p>{getReadableError(indexQuery.error)}</p>
          <div className="search-state-actions">
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void indexQuery.refetch()}
            >
              Réessayer
            </button>
            <Link className="button button-ghost" to="/parametres">
              Ouvrir les paramètres
            </Link>
          </div>
        </div>
      ) : (
        <IndexNotice
          state={indexQuery.data.state}
          indexedNodes={indexQuery.data.indexed_nodes}
          totalNodes={indexQuery.data.total_nodes}
          embeddingReady={embeddingIsReady}
          embeddingError={indexQuery.data.embedding.error}
        />
      )}

      {readinessQuery.isPending ? (
        <div className="loading-state search-loading rag-readiness-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Vérification du modèle de génération…
        </div>
      ) : readinessQuery.isError ? (
        <div className="empty-state error-state search-state" role="alert">
          <h2>Impossible de vérifier Ollama</h2>
          <p>{getReadableError(readinessQuery.error)}</p>
          <div className="search-state-actions">
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void readinessQuery.refetch()}
            >
              Réessayer
            </button>
            <Link className="button button-ghost" to="/parametres">
              Ouvrir les paramètres
            </Link>
          </div>
        </div>
      ) : (
        <GenerationNotice
          ollamaAvailable={readinessQuery.data.ollama.available}
          modelAvailable={readinessQuery.data.ollama.model_available}
          model={readinessQuery.data.ollama.configured_model}
          error={readinessQuery.data.ollama.error}
        />
      )}

      {ragMutation.isPending ? (
        <div className="loading-state search-loading rag-generation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          <div>
            <strong>Recherche des connaissances et génération locale…</strong>
            <p>Le modèle peut prendre quelques instants sur cette machine.</p>
          </div>
        </div>
      ) : ragMutation.isError ? (
        <div className="empty-state error-state search-state" role="alert">
          <h2>La réponse n’a pas pu être générée</h2>
          <p>{getReadableError(ragMutation.error)}</p>
          <div className="search-state-actions">
            <button
              className="button button-secondary"
              type="button"
              disabled={!ragMutation.variables || !ragCanRun}
              onClick={() => {
                if (ragMutation.variables) {
                  ragMutation.mutate(ragMutation.variables);
                }
              }}
            >
              Réessayer
            </button>
            <Link className="button button-ghost" to="/parametres">
              Vérifier les paramètres
            </Link>
          </div>
        </div>
      ) : response ? (
        <>
          <section className="search-results rag-response" aria-labelledby="rag-answer-title">
            <div className="search-results-heading" aria-live="polite">
              <div>
                <p className="eyebrow">
                  {response.mode === "brain_only"
                    ? "Second cerveau uniquement"
                    : "Second cerveau + modèle"}
                </p>
                <h2 id="rag-answer-title">Réponse</h2>
              </div>
              <p>à « {response.question} »</p>
            </div>

            <article
              className={`panel rag-answer-panel${response.insufficient_context ? " is-insufficient" : ""}`}
            >
              <div className="rag-answer-heading">
                <h3>Informations du second cerveau</h3>
                {response.insufficient_context ? (
                  <span className="rag-insufficient-badge">
                    Contexte insuffisant
                  </span>
                ) : (
                  <span className="rag-grounded-badge">Réponse sourcée</span>
                )}
              </div>
              <CitedAnswer
                text={response.answer}
                usedKnowledge={response.used_knowledge}
                returnPath={returnPath}
              />

              {response.mode === "brain_plus_model" &&
              response.model_additions?.trim() ? (
                <section
                  className="rag-model-additions"
                  aria-labelledby="rag-model-additions-title"
                >
                  <h3 id="rag-model-additions-title">
                    Complément des connaissances générales du modèle
                  </h3>
                  <p>{response.model_additions}</p>
                  <small>
                    Cette partie ne provient pas de votre second cerveau.
                  </small>
                </section>
              ) : null}

              <footer className="rag-answer-footer">
                <span>Modèle : {response.generation_model}</span>
                <span>Réponse : {response.request_id}</span>
              </footer>
            </article>

            <RagTimingsDetails timings={response.timings} />
          </section>

          <section
            className="search-results rag-provenance"
            aria-labelledby="rag-provenance-title"
          >
            <div className="search-results-heading">
              <div>
                <p className="eyebrow">Traçabilité du retrieval</p>
                <h2 id="rag-provenance-title">Connaissances utilisées</h2>
              </div>
              <p>
                {response.retrieved_knowledge.length.toLocaleString("fr-FR")} récupérée
                {response.retrieved_knowledge.length > 1 ? "s" : ""} ·{" "}
                {providedKnowledgeCount.toLocaleString("fr-FR")} fournie
                {providedKnowledgeCount > 1 ? "s" : ""} ·{" "}
                {response.used_knowledge.length.toLocaleString("fr-FR")} citée
                {response.used_knowledge.length > 1 ? "s" : ""}
              </p>
            </div>

            <ol className="rag-retrieval-flow" aria-label="Étapes de la réponse">
              <li>
                <span>1</span>
                Question
              </li>
              <li>
                <span>2</span>
                {response.retrieved_knowledge.length} connaissances récupérées
              </li>
              <li>
                <span>3</span>
                {providedKnowledgeCount} connaissances dans le contexte
              </li>
              <li>
                <span>4</span>
                {response.used_knowledge.length} connaissances citées
              </li>
            </ol>

            {response.used_knowledge.length === 0 ? (
              <div className="empty-state compact-empty-state rag-no-used-knowledge">
                <p>
                  Aucune connaissance n’a été citée dans cette réponse. Le détail
                  du retrieval reste disponible ci-dessous.
                </p>
              </div>
            ) : (
              <ol className="semantic-result-list rag-used-list">
                {response.used_knowledge.map((item, index) => (
                  <RagKnowledgeCard
                    item={item}
                    rank={index + 1}
                    returnPath={returnPath}
                    key={`${item.context_id ?? "used"}-${item.knowledge_node.id}`}
                  />
                ))}
              </ol>
            )}

            <details className="rag-retrieval-details">
              <summary>
                Afficher tous les résultats de la recherche sémantique
              </summary>
              {response.retrieved_knowledge.length === 0 ? (
                <div className="empty-state compact-empty-state">
                  <p>L’index n’a retourné aucune connaissance.</p>
                </div>
              ) : (
                <ol className="semantic-result-list">
                  {response.retrieved_knowledge.map((item, index) => (
                    <RagKnowledgeCard
                      item={item}
                      rank={index + 1}
                      returnPath={returnPath}
                      key={`${item.context_id ?? "retrieved"}-${item.knowledge_node.id}`}
                    />
                  ))}
                </ol>
              )}
            </details>
          </section>
        </>
      ) : submittedQuestion.length === 1 ? (
        <div className="empty-state search-state" role="status">
          <h2>Requête trop courte</h2>
          <p>Saisissez au moins deux caractères pour poser une question.</p>
        </div>
      ) : !submittedQuestion ? (
        <div className="empty-state search-state search-initial-state">
          <div className="empty-icon" aria-hidden="true">
            ⌕
          </div>
          <h2>Interrogez vos connaissances</h2>
          <p>
            Le mode par défaut refuse de compléter avec des connaissances
            générales lorsque votre second cerveau ne suffit pas.
          </p>
        </div>
      ) : (
        <div className="empty-state search-state" role="status">
          <h2>Question prête</h2>
          <p>Appuyez sur Envoyer pour générer une nouvelle réponse sourcée.</p>
        </div>
      )}
    </section>
  );
}

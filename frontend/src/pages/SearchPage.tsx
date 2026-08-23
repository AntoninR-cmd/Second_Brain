import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  getReadableError,
  getVectorIndexStatus,
  searchSemantically,
} from "../api/client";
import type {
  KnowledgeEvidence,
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

function canSearchIndex(state: VectorIndexState, indexedNodes: number): boolean {
  return indexedNodes > 0 && (state === "ready" || state === "stale");
}

function IndexNotice({
  state,
  indexedNodes,
  totalNodes,
}: {
  state: VectorIndexState;
  indexedNodes: number;
  totalNodes: number;
}) {
  if (state === "ready") {
    return null;
  }

  const isBlocking = !canSearchIndex(state, indexedNodes);
  let message = "L’index sémantique doit être préparé avant la recherche.";

  if (state === "empty") {
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
        <strong>{INDEX_STATE_LABELS[state]}</strong>
        <p>{message}</p>
      </div>
      <Link className="button button-secondary" to="/parametres">
        Ouvrir les paramètres
      </Link>
    </div>
  );
}

export function SearchPage() {
  const [searchParameters, setSearchParameters] = useSearchParams();
  const submittedQuery = searchParameters.get("q")?.trim() ?? "";
  const [draftQuery, setDraftQuery] = useState(submittedQuery);

  useEffect(() => {
    setDraftQuery(submittedQuery);
  }, [submittedQuery]);

  const indexQuery = useQuery({
    queryKey: ["vector-index", "status"],
    queryFn: getVectorIndexStatus,
    refetchInterval: (query) =>
      query.state.data?.state === "building" ? 1_500 : 15_000,
  });

  const indexCanBeSearched = indexQuery.data
    ? canSearchIndex(indexQuery.data.state, indexQuery.data.indexed_nodes)
    : false;
  const searchQuery = useQuery({
    queryKey: ["semantic-search", submittedQuery],
    queryFn: () => searchSemantically({ query: submittedQuery }),
    enabled: submittedQuery.length >= 2 && indexCanBeSearched,
  });

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = draftQuery.trim();

    if (!normalizedQuery) {
      setSearchParameters({});
      return;
    }

    if (normalizedQuery === submittedQuery && indexCanBeSearched) {
      void searchQuery.refetch();
      return;
    }

    setSearchParameters({ q: normalizedQuery });
  }

  const resultItems = searchQuery.data?.items ?? [];

  return (
    <section className="page search-page">
      <header className="page-header search-header">
        <div>
          <p className="eyebrow">Mémoire sémantique locale</p>
          <h1>Recherche dans mon cerveau</h1>
          <p className="page-introduction">
            Retrouvez des connaissances par leur sens, même avec des mots
            différents. Aucune réponse n’est générée par un LLM.
          </p>
        </div>
      </header>

      <form className="panel semantic-search-form" role="search" onSubmit={submitSearch}>
        <label htmlFor="semantic-query">Question ou idée à retrouver</label>
        <div className="semantic-search-controls">
          <input
            id="semantic-query"
            name="q"
            type="search"
            value={draftQuery}
            placeholder="Comment empêcher une peinture de se décoller d’un pare-chocs ?"
            autoComplete="off"
            aria-describedby="semantic-search-help"
            onChange={(event) => setDraftQuery(event.target.value)}
          />
          <button
            className="button button-primary"
            type="submit"
            disabled={
              draftQuery.trim().length < 2 ||
              indexQuery.isPending ||
              !indexCanBeSearched ||
              searchQuery.isFetching
            }
          >
            {searchQuery.isFetching ? (
              <>
                <span className="spinner spinner-light" aria-hidden="true" />
                Recherche…
              </>
            ) : (
              "Rechercher"
            )}
          </button>
        </div>
        <p id="semantic-search-help">
          La requête est encodée localement puis comparée à l’index Qdrant.
        </p>
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
        />
      )}

      {submittedQuery.length >= 2 && indexCanBeSearched ? (
        <section className="search-results" aria-labelledby="search-results-title">
          <div className="search-results-heading" aria-live="polite">
            <div>
              <p className="eyebrow">Résultats sémantiques</p>
              <h2 id="search-results-title">
                {searchQuery.isFetching && !searchQuery.data
                  ? "Recherche en cours…"
                  : `${resultItems.length.toLocaleString("fr-FR")} résultat${resultItems.length > 1 ? "s" : ""}`}
              </h2>
            </div>
            <p>pour « {submittedQuery} »</p>
          </div>

          {searchQuery.isPending ? (
            <div className="loading-state search-loading" role="status">
              <span className="spinner" aria-hidden="true" />
              Comparaison des connaissances…
            </div>
          ) : searchQuery.isError ? (
            <div className="empty-state error-state search-state" role="alert">
              <h2>La recherche n’a pas abouti</h2>
              <p>{getReadableError(searchQuery.error)}</p>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => void searchQuery.refetch()}
              >
                Réessayer
              </button>
            </div>
          ) : resultItems.length === 0 ? (
            <div className="empty-state search-state">
              <h2>Aucun résultat</h2>
              <p>
                L’index n’a trouvé aucune connaissance pour cette requête.
                Essayez une formulation plus générale.
              </p>
            </div>
          ) : (
            <ol className="semantic-result-list">
              {resultItems.map((result, index) => {
                const node = result.knowledge_node;
                const primaryEvidence = result.evidences[0];
                const nodeHref = `/connaissances/${encodeURIComponent(node.id)}`;

                return (
                  <li className="semantic-result-card" key={node.id}>
                    <div className="semantic-result-rank" aria-hidden="true">
                      {index + 1}
                    </div>
                    <article>
                      <div className="semantic-result-heading">
                        <h3>
                          <Link
                            to={nodeHref}
                            state={{
                              fromSearch: `/recherche?q=${encodeURIComponent(submittedQuery)}`,
                            }}
                          >
                            {node.title}
                          </Link>
                        </h3>
                        <span className="similarity-score">
                          similarité : {scoreFormatter.format(result.score)}
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
                        <span>{getSourceTypeLabel(result.source.type)}</span>
                        <Link to={`/sources/${result.source.id}`}>
                          {result.source.title}
                        </Link>
                        {result.source.author ? (
                          <span>par {result.source.author}</span>
                        ) : null}
                      </div>

                      {primaryEvidence ? (
                        <div className="semantic-result-evidence">
                          <strong>{getEvidenceLocator(primaryEvidence)}</strong>
                          <blockquote>{primaryEvidence.original_excerpt}</blockquote>
                        </div>
                      ) : null}

                      <Link
                        className="knowledge-open-link"
                        to={nodeHref}
                        state={{
                          fromSearch: `/recherche?q=${encodeURIComponent(submittedQuery)}`,
                        }}
                      >
                        Ouvrir la connaissance
                        <span aria-hidden="true">→</span>
                      </Link>
                    </article>
                  </li>
                );
              })}
            </ol>
          )}
        </section>
      ) : submittedQuery.length === 1 ? (
        <div className="empty-state search-state" role="status">
          <h2>Requête trop courte</h2>
          <p>Saisissez au moins deux caractères pour lancer la recherche.</p>
        </div>
      ) : !submittedQuery ? (
        <div className="empty-state search-state search-initial-state">
          <div className="empty-icon" aria-hidden="true">
            ⌕
          </div>
          <h2>Interrogez vos connaissances</h2>
          <p>
            Décrivez ce que vous cherchez : la correspondance repose sur le
            sens, pas seulement sur les mots exacts.
          </p>
        </div>
      ) : null}
    </section>
  );
}

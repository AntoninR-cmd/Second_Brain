import { useInfiniteQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { getReadableError, getSources } from "../api/client";
import {
  formatDate,
  getProcessingStatusLabel,
  getSourceTypeLabel,
} from "../utils/sourcePresentation";

export function SourcesPage() {
  const sourcesQuery = useInfiniteQuery({
    queryKey: ["sources"],
    queryFn: ({ pageParam }) => getSources(pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });

  const sources = sourcesQuery.data?.pages.flatMap((page) => page.items) ?? [];

  return (
    <section className="page">
      <header className="page-header dashboard-header">
        <div>
          <p className="eyebrow">Bibliothèque locale</p>
          <h1>Sources</h1>
          <p className="page-introduction">
            Consultez les notes et fichiers conservés dans votre Second Brain.
          </p>
        </div>
        <Link className="button button-primary" to="/ajouter">
          <span aria-hidden="true">＋</span>
          Ajouter une source
        </Link>
      </header>

      <section className="panel sources-panel" aria-labelledby="sources-title">
        <div className="panel-header">
          <div>
            <h2 id="sources-title">Toutes les sources</h2>
            <p>
              Notes manuelles et fichiers importés, du plus récent au plus ancien.
            </p>
          </div>
        </div>

        {sourcesQuery.isPending ? (
          <div className="loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Chargement des sources…
          </div>
        ) : sourcesQuery.isError ? (
          <div className="empty-state error-state" role="alert">
            <h3>Impossible de charger les sources</h3>
            <p>{getReadableError(sourcesQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void sourcesQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        ) : sources.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm8 2v5h4L14 4Zm-2 8a1 1 0 0 0-1 1v2H9a1 1 0 1 0 0 2h2v2a1 1 0 1 0 2 0v-2h2a1 1 0 1 0 0-2h-2v-2a1 1 0 0 0-1-1Z" />
              </svg>
            </div>
            <h3>Aucune source enregistrée</h3>
            <p>Ajoutez une note ou importez un fichier SRT ou TXT.</p>
            <Link className="button button-secondary" to="/ajouter">
              Ajouter une source
            </Link>
          </div>
        ) : (
          <>
            <div className="source-table-wrapper">
              <table className="source-table">
                <thead>
                  <tr>
                    <th scope="col">Titre</th>
                    <th scope="col">Type</th>
                    <th scope="col">Auteur</th>
                    <th scope="col">Ajoutée le</th>
                    <th scope="col">Statut</th>
                  </tr>
                </thead>
                <tbody>
                  {sources.map((source) => (
                    <tr key={source.id}>
                      <td>
                        <Link
                          className="source-title-link"
                          to={`/sources/${source.id}`}
                        >
                          {source.title}
                        </Link>
                        {source.original_filename ? (
                          <span className="source-filename">
                            {source.original_filename}
                          </span>
                        ) : null}
                      </td>
                      <td>
                        <span className={`source-type type-${source.type}`}>
                          {getSourceTypeLabel(source.type)}
                        </span>
                      </td>
                      <td>{source.author ?? "—"}</td>
                      <td>{formatDate(source.created_at)}</td>
                      <td>
                        <span className="ready-label">
                          {getProcessingStatusLabel(source.processing_status)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {sourcesQuery.hasNextPage ? (
              <div className="list-actions">
                <button
                  className="button button-secondary"
                  type="button"
                  disabled={sourcesQuery.isFetchingNextPage}
                  onClick={() => void sourcesQuery.fetchNextPage()}
                >
                  {sourcesQuery.isFetchingNextPage ? (
                    <>
                      <span className="spinner" aria-hidden="true" />
                      Chargement…
                    </>
                  ) : (
                    "Afficher plus"
                  )}
                </button>
              </div>
            ) : null}
          </>
        )}
      </section>
    </section>
  );
}

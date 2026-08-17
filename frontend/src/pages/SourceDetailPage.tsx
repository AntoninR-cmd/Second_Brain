import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { Link, useLocation, useParams } from "react-router-dom";

import {
  getReadableError,
  getSource,
  getSourceSegments,
} from "../api/client";
import {
  formatDate,
  formatSrtTimestamp,
  getProcessingStatusLabel,
  getSourceTypeLabel,
} from "../utils/sourcePresentation";

interface SourceDetailLocationState {
  flash?: string;
}

export function SourceDetailPage() {
  const { sourceId } = useParams();
  const location = useLocation();
  const flash = (location.state as SourceDetailLocationState | null)?.flash;

  const sourceQuery = useQuery({
    queryKey: ["sources", sourceId],
    queryFn: () => getSource(sourceId ?? ""),
    enabled: Boolean(sourceId),
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
  const segmentTotal = source.segment_count;

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

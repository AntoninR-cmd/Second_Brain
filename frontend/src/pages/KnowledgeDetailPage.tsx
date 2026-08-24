import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useParams } from "react-router-dom";

import { getKnowledgeNode, getReadableError } from "../api/client";
import type { KnowledgeEvidence } from "../api/types";
import {
  formatDate,
  formatSrtTimestamp,
  getSourceTypeLabel,
} from "../utils/sourcePresentation";

interface KnowledgeDetailLocationState {
  fromSearch?: string;
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

export function KnowledgeDetailPage() {
  const { nodeId } = useParams();
  const location = useLocation();
  const requestedSearchReturn = (
    location.state as KnowledgeDetailLocationState | null
  )?.fromSearch;
  const searchReturn = requestedSearchReturn?.startsWith("/recherche")
    ? requestedSearchReturn
    : null;
  const nodeQuery = useQuery({
    queryKey: ["knowledge-nodes", nodeId],
    queryFn: () => getKnowledgeNode(nodeId ?? ""),
    enabled: Boolean(nodeId),
  });

  if (!nodeId) {
    return (
      <section className="page narrow-page">
        <p className="eyebrow">Connaissance introuvable</p>
        <h1>Identifiant de connaissance manquant</h1>
        <Link className="button button-primary detail-back-button" to="/sources">
          Revenir aux sources
        </Link>
      </section>
    );
  }

  if (nodeQuery.isPending) {
    return (
      <section className="page">
        <div className="loading-state page-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Chargement de la connaissance…
        </div>
      </section>
    );
  }

  if (nodeQuery.isError) {
    return (
      <section className="page narrow-page">
        <p className="eyebrow">Connaissance indisponible</p>
        <h1>Impossible d’ouvrir cette connaissance</h1>
        <p className="page-introduction">
          {getReadableError(nodeQuery.error)}
        </p>
        <div className="detail-error-actions">
          <Link className="button button-ghost" to={searchReturn ?? "/sources"}>
            {searchReturn ? "Revenir à la recherche" : "Revenir aux sources"}
          </Link>
          <button
            className="button button-primary"
            type="button"
            onClick={() => void nodeQuery.refetch()}
          >
            Réessayer
          </button>
        </div>
      </section>
    );
  }

  const node = nodeQuery.data;

  return (
    <section className="page knowledge-detail-page">
      <Link
        className="back-link"
        to={searchReturn ?? `/sources/${node.source.id}`}
      >
        <span aria-hidden="true">←</span>
        {searchReturn ? "Revenir à la recherche" : "Revenir à la source"}
      </Link>

      <header className="page-header knowledge-detail-header">
        <div>
          <p className="eyebrow">Connaissance atomique</p>
          <h1>{node.title}</h1>
          <p className="page-introduction">
            Extraite de « {node.source.title} » · créée le {formatDate(node.created_at)}
          </p>
        </div>
      </header>

      <article className="panel knowledge-content-panel">
        <p className="knowledge-content">{node.content}</p>
        {node.tags.length > 0 ? (
          <ul className="tag-list" aria-label="Tags automatiques">
            {node.tags.map((tag) => (
              <li key={tag}>#{tag}</li>
            ))}
          </ul>
        ) : null}
      </article>

      <section className="panel detail-panel" aria-labelledby="knowledge-source-title">
        <div className="panel-header">
          <div>
            <h2 id="knowledge-source-title">Source d’origine</h2>
            <p>Le contenu original reste inchangé et consultable.</p>
          </div>
        </div>
        <dl className="metadata-list">
          <div>
            <dt>Titre</dt>
            <dd>
              <Link className="source-title-link" to={`/sources/${node.source.id}`}>
                {node.source.title}
              </Link>
            </dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{getSourceTypeLabel(node.source.type)}</dd>
          </div>
          <div>
            <dt>Auteur</dt>
            <dd>{node.source.author ?? "Non renseigné"}</dd>
          </div>
          {node.source.original_filename ? (
            <div className="metadata-wide">
              <dt>Fichier original</dt>
              <dd>{node.source.original_filename}</dd>
            </div>
          ) : null}
          {node.source.original_file_path ? (
            <div className="metadata-wide">
              <dt>Copie locale</dt>
              <dd>
                <code>{node.source.original_file_path}</code>
              </dd>
            </div>
          ) : null}
        </dl>
      </section>

      <section className="panel detail-panel" aria-labelledby="evidence-title">
        <div className="panel-header">
          <div>
            <h2 id="evidence-title">Passages justificatifs</h2>
            <p>
              {node.evidences.length.toLocaleString("fr-FR")} passage
              {node.evidences.length > 1 ? "s" : ""} utilisé
              {node.evidences.length > 1 ? "s" : ""} pour cette connaissance.
            </p>
          </div>
        </div>

        {node.evidences.length === 0 ? (
          <div className="empty-state compact-empty-state">
            <p>Aucun passage justificatif n’est disponible.</p>
          </div>
        ) : (
          <ol className="evidence-list">
            {node.evidences.map((evidence) => (
              <li className="evidence-card" key={evidence.id}>
                <div className="evidence-heading">
                  <strong>Passage #{evidence.passage_index}</strong>
                  <span>{getEvidenceLocator(evidence)}</span>
                </div>
                <blockquote>{evidence.original_excerpt}</blockquote>
              </li>
            ))}
          </ol>
        )}
      </section>
    </section>
  );
}

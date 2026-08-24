import { Link } from "react-router-dom";

import type {
  BrainClusterDetail,
  KnowledgeNodeDetail,
} from "../../api/types";
import { getKnowledgeEvidenceLocator } from "../../utils/sourcePresentation";

interface BrainDetailsPanelProps {
  cluster: BrainClusterDetail | null;
  knowledge: KnowledgeNodeDetail | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onEnterCluster: (cluster: BrainClusterDetail) => void;
  onSelectChild: (clusterId: string) => void;
}

export function BrainDetailsPanel({
  cluster,
  knowledge,
  loading,
  error,
  onClose,
  onEnterCluster,
  onSelectChild,
}: BrainDetailsPanelProps) {
  const title = cluster?.label ?? knowledge?.title ?? "Détail";

  return (
    <aside
      className="brain-details-panel"
      aria-labelledby="brain-details-title"
      aria-live="polite"
    >
      <header className="brain-details-header">
        <div>
          <p className="brain-panel-kicker">
            {cluster ? "Thème" : knowledge ? "Connaissance" : "Sélection"}
          </p>
          <h2 id="brain-details-title">{title}</h2>
        </div>
        <button
          className="brain-icon-button"
          type="button"
          aria-label="Fermer le panneau de détail"
          onClick={onClose}
        >
          ×
        </button>
      </header>

      {loading ? (
        <div className="brain-panel-state" role="status">
          <span className="spinner spinner-light" aria-hidden="true" />
          Chargement du détail…
        </div>
      ) : null}

      {error ? (
        <div className="brain-panel-state is-error" role="alert">
          {error}
        </div>
      ) : null}

      {cluster && !loading ? (
        <div className="brain-panel-content">
          <p className="brain-cluster-description">
            {cluster.description ??
              "Ce thème a été construit automatiquement à partir des proximités sémantiques."}
          </p>
          <dl className="brain-panel-metrics">
            <div>
              <dt>Connaissances</dt>
              <dd>{cluster.member_count.toLocaleString("fr-FR")}</dd>
            </div>
            <div>
              <dt>Sous-thèmes</dt>
              <dd>{cluster.child_count.toLocaleString("fr-FR")}</dd>
            </div>
          </dl>

          {cluster.children.length > 0 ? (
            <section className="brain-panel-section">
              <h3>Sous-thèmes</h3>
              <ul className="brain-child-list">
                {cluster.children.map((child) => (
                  <li key={child.id}>
                    <button type="button" onClick={() => onSelectChild(child.id)}>
                      <span>{child.label}</span>
                      <small>{child.member_count} connaissances</small>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          <button
            className="button brain-enter-button"
            type="button"
            onClick={() => onEnterCluster(cluster)}
          >
            Entrer dans ce thème
            <span aria-hidden="true">→</span>
          </button>
        </div>
      ) : null}

      {knowledge && !loading ? (
        <div className="brain-panel-content">
          <p className="brain-knowledge-content">{knowledge.content}</p>

          {knowledge.tags.length > 0 ? (
            <ul className="brain-tag-list" aria-label="Tags de la connaissance">
              {knowledge.tags.map((tag) => (
                <li key={tag}>#{tag}</li>
              ))}
            </ul>
          ) : null}

          <section className="brain-panel-section">
            <h3>Source</h3>
            <p className="brain-source-name">{knowledge.source.title}</p>
            {knowledge.source.author ? (
              <p className="brain-source-author">{knowledge.source.author}</p>
            ) : null}
          </section>

          <section className="brain-panel-section">
            <h3>
              Preuves
              <span className="brain-count-badge">{knowledge.evidences.length}</span>
            </h3>
            {knowledge.evidences.length > 0 ? (
              <ol className="brain-evidence-list">
                {knowledge.evidences.map((evidence) => (
                  <li key={evidence.id}>
                    <strong>{getKnowledgeEvidenceLocator(evidence)}</strong>
                    <blockquote>{evidence.original_excerpt}</blockquote>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="brain-panel-muted">Aucune preuve disponible.</p>
            )}
          </section>

          <Link
            className="button brain-enter-button"
            to={`/connaissances/${knowledge.id}`}
            state={{ fromBrain: "/cerveau" }}
          >
            Ouvrir la connaissance
            <span aria-hidden="true">↗</span>
          </Link>
        </div>
      ) : null}

      {!cluster && !knowledge && !loading && !error ? (
        <div className="brain-panel-state">
          Sélectionnez un thème ou une connaissance sur la carte.
        </div>
      ) : null}
    </aside>
  );
}

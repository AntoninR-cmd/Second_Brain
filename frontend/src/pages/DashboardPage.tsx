import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { getDashboard, getHealth, getReadableError } from "../api/client";
import type { SourceDetail } from "../api/types";
import {
  formatDate,
  getSourceTypeLabel,
  textExcerpt,
} from "../utils/sourcePresentation";

interface DashboardLocationState {
  flash?: string;
}

function RecentSource({ source }: { source: SourceDetail }) {
  return (
    <li className="recent-note">
      <div className="note-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24">
          <path d="M6 2h8.6L20 7.4V20a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm8 2v5h4v-.8L13.8 4H14Zm-6 9a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2H8Zm0 4a1 1 0 1 0 0 2h5a1 1 0 1 0 0-2H8Z" />
        </svg>
      </div>
      <div className="recent-note-content">
        <span className={`source-type type-${source.type}`}>
          {getSourceTypeLabel(source.type)}
        </span>
        <h3>
          <Link className="source-title-link" to={`/sources/${source.id}`}>
            {source.title}
          </Link>
        </h3>
        <p className="note-excerpt">{textExcerpt(source.raw_text)}</p>
        <p className="note-metadata">
          {source.author ? `${source.author} · ` : ""}
          Ajoutée le {formatDate(source.created_at)}
        </p>
      </div>
      <span className="ready-label">Prête</span>
    </li>
  );
}

export function DashboardPage() {
  const location = useLocation();
  const initialFlash = (location.state as DashboardLocationState | null)?.flash;
  const [flash, setFlash] = useState(initialFlash ?? null);

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  const dashboardQuery = useQuery({
    queryKey: ["dashboard"],
    queryFn: getDashboard,
  });

  const apiIsHealthy =
    healthQuery.isSuccess && healthQuery.data.database === "ok";

  return (
    <section className="page">
      <header className="page-header dashboard-header">
        <div>
          <p className="eyebrow">Vue d’ensemble</p>
          <h1>Dashboard</h1>
          <p className="page-introduction">
            Retrouvez les sources conservées dans votre Second Brain.
          </p>
        </div>
        <Link className="button button-primary" to="/ajouter">
          <span aria-hidden="true">＋</span>
          Ajouter une source
        </Link>
      </header>

      {flash ? (
        <div className="alert alert-success" role="status">
          <span aria-hidden="true">✓</span>
          <p>{flash}</p>
          <button
            className="alert-close"
            type="button"
            aria-label="Fermer le message"
            onClick={() => setFlash(null)}
          >
            ×
          </button>
        </div>
      ) : null}

      <section className="stat-grid" aria-label="État du Second Brain">
        <article className="stat-card">
          <div className="stat-card-heading">
            <span>Sources enregistrées</span>
            <span className="stat-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm8 2v5h4L14 4Z" />
              </svg>
            </span>
          </div>
          {dashboardQuery.isPending ? (
            <span className="stat-value skeleton-text" aria-label="Chargement">
              —
            </span>
          ) : dashboardQuery.isError ? (
            <span className="stat-value">—</span>
          ) : (
            <strong className="stat-value">
              {dashboardQuery.data.source_count.toLocaleString("fr-FR")}
            </strong>
          )}
          <p>Persistées dans SQLite</p>
        </article>

        <article className="stat-card">
          <div className="stat-card-heading">
            <span>État du système</span>
            <span
              className={`status-dot${apiIsHealthy ? " is-online" : ""}`}
              aria-hidden="true"
            />
          </div>
          <strong className="system-status">
            {healthQuery.isPending
              ? "Vérification…"
              : apiIsHealthy
                ? "Opérationnel"
                : "Indisponible"}
          </strong>
          <p>
            {apiIsHealthy
              ? "API et base SQLite accessibles"
              : healthQuery.isPending
                ? "Connexion à l’API locale"
                : "Vérifiez que le backend est démarré"}
          </p>
        </article>
      </section>

      {healthQuery.isError ? (
        <div className="alert alert-error compact-alert" role="alert">
          <p>{getReadableError(healthQuery.error)}</p>
          <button
            className="text-button"
            type="button"
            onClick={() => void healthQuery.refetch()}
          >
            Réessayer
          </button>
        </div>
      ) : null}

      <section className="panel recent-panel" aria-labelledby="recent-title">
        <div className="panel-header">
          <div>
            <h2 id="recent-title">Sources récentes</h2>
            <p>Les cinq derniers ajouts enregistrés.</p>
          </div>
        </div>

        {dashboardQuery.isPending ? (
          <div className="loading-state" role="status">
            <span className="spinner" aria-hidden="true" />
            Chargement des sources…
          </div>
        ) : dashboardQuery.isError ? (
          <div className="empty-state error-state" role="alert">
            <h3>Impossible de charger les sources</h3>
            <p>{getReadableError(dashboardQuery.error)}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void dashboardQuery.refetch()}
            >
              Réessayer
            </button>
          </div>
        ) : dashboardQuery.data.recent_sources.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm8 2v5h4L14 4Zm-2 8a1 1 0 0 0-1 1v2H9a1 1 0 1 0 0 2h2v2a1 1 0 1 0 2 0v-2h2a1 1 0 1 0 0-2h-2v-2a1 1 0 0 0-1-1Z" />
              </svg>
            </div>
            <h3>Votre Second Brain est vide</h3>
            <p>Ajoutez une note ou importez un fichier SRT ou TXT pour commencer.</p>
            <Link className="button button-secondary" to="/ajouter">
              Ajouter une source
            </Link>
          </div>
        ) : (
          <ul className="recent-list">
            {dashboardQuery.data.recent_sources.map((source) => (
              <RecentSource key={source.id} source={source} />
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}

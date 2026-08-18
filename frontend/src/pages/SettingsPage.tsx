import { useQuery } from "@tanstack/react-query";

import { getReadableError, getSystemReadiness } from "../api/client";

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

export function SettingsPage() {
  const readinessQuery = useQuery({
    queryKey: ["system", "readiness"],
    queryFn: getSystemReadiness,
    refetchInterval: 30_000,
  });

  return (
    <section className="page narrow-page">
      <header className="page-header settings-header">
        <div>
          <p className="eyebrow">Configuration locale</p>
          <h1>Paramètres</h1>
          <p className="page-introduction">
            Vérifiez la connexion entre FastAPI et Ollama. Le navigateur ne
            contacte jamais Ollama directement.
          </p>
        </div>
        <button
          className="button button-secondary"
          type="button"
          disabled={readinessQuery.isFetching}
          onClick={() => void readinessQuery.refetch()}
        >
          {readinessQuery.isFetching ? (
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
            Vérification d’Ollama…
          </div>
        </section>
      ) : readinessQuery.isError ? (
        <section className="panel settings-panel">
          <div className="empty-state error-state" role="alert">
            <h2>Impossible de vérifier le système</h2>
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
        <section className="panel settings-panel" aria-labelledby="ollama-title">
          <div className="panel-header">
            <div>
              <h2 id="ollama-title">Génération locale avec Ollama</h2>
              <p>Diagnostic fourni par le backend Second Brain.</p>
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

          <div className="settings-guidance">
            <h3>Aucun téléchargement automatique</h3>
            <p>
              Second Brain ne télécharge jamais de modèle seul. Installez Ollama
              et le modèle configuré explicitement avant de lancer une analyse.
            </p>
          </div>
        </section>
      )}
    </section>
  );
}

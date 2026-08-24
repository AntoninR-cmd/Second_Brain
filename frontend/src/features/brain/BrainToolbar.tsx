export interface BrainPerformanceMetrics {
  apiMs: number | null;
  graphologyMs: number | null;
  firstPaintMs: number | null;
  navigationMs: number | null;
  hoverMs: number | null;
}

interface BrainToolbarProps {
  viewLabel: string;
  metrics: BrainPerformanceMetrics;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onRecenter: () => void;
  onGlobalView: () => void;
  onFullscreen: () => void;
}

function formatMetric(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)} ms`;
}

export function BrainToolbar({
  viewLabel,
  metrics,
  onZoomIn,
  onZoomOut,
  onRecenter,
  onGlobalView,
  onFullscreen,
}: BrainToolbarProps) {
  return (
    <div className="brain-toolbar" aria-label="Contrôles de la carte">
      <div className="brain-view-indicator" aria-live="polite">
        <span>Vue</span>
        <strong>{viewLabel}</strong>
      </div>

      <div className="brain-camera-controls">
        <button type="button" aria-label="Zoomer" onClick={onZoomIn}>
          +
        </button>
        <button type="button" aria-label="Dézoomer" onClick={onZoomOut}>
          −
        </button>
        <button type="button" aria-label="Recentrer le niveau courant" onClick={onRecenter}>
          ◎
        </button>
        <button type="button" aria-label="Revenir à la vue globale" onClick={onGlobalView}>
          ⌂
        </button>
        <button type="button" aria-label="Afficher le cerveau en plein écran" onClick={onFullscreen}>
          ⛶
        </button>
      </div>

      <details className="brain-performance">
        <summary>Performances</summary>
        <dl>
          <div>
            <dt>API</dt>
            <dd>{formatMetric(metrics.apiMs)}</dd>
          </div>
          <div>
            <dt>Graphology</dt>
            <dd>{formatMetric(metrics.graphologyMs)}</dd>
          </div>
          <div>
            <dt>Première peinture</dt>
            <dd>{formatMetric(metrics.firstPaintMs)}</dd>
          </div>
          <div>
            <dt>Navigation</dt>
            <dd>{formatMetric(metrics.navigationMs)}</dd>
          </div>
          <div>
            <dt>Hover</dt>
            <dd>{formatMetric(metrics.hoverMs)}</dd>
          </div>
        </dl>
      </details>
    </div>
  );
}

export function BrainLegend() {
  return (
    <div className="brain-legend" aria-label="Légende du cerveau">
      <span>
        <i className="is-cluster" aria-hidden="true" /> Domaine ou thème
      </span>
      <span>
        <i className="is-knowledge" aria-hidden="true" /> Connaissance
      </span>
      <span>
        <i className="is-edge" aria-hidden="true" /> Proximité sémantique
      </span>
    </div>
  );
}

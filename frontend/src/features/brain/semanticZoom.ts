import type { BrainGraphNodeKind } from "./contracts";
import type { SemanticZoomTier } from "./types";

/**
 * Sigma's camera ratio grows when zooming out and shrinks when zooming in.
 * The hysteresis-free pure function is intentionally isolated so the canvas can
 * debounce/API-load separately without coupling navigation to the renderer.
 */
export const SEMANTIC_ZOOM_THRESHOLDS = Object.freeze({
  domainsMinimumRatio: 1.2,
  knowledgeMaximumRatio: 0.52,
});

export function semanticZoomTierForRatio(
  cameraRatio: number,
): SemanticZoomTier {
  const safeRatio = Number.isFinite(cameraRatio)
    ? Math.max(0.01, cameraRatio)
    : 1;
  if (safeRatio >= SEMANTIC_ZOOM_THRESHOLDS.domainsMinimumRatio) {
    return "domains";
  }
  if (safeRatio <= SEMANTIC_ZOOM_THRESHOLDS.knowledgeMaximumRatio) {
    return "knowledge";
  }
  return "themes";
}

export function nodeKindVisibleAtZoom(
  kind: BrainGraphNodeKind,
  tier: SemanticZoomTier,
  graphContainsClusters: boolean,
): boolean {
  // A leaf payload contains only KnowledgeNodes: it must never become an empty
  // canvas merely because the user zoomed out inside that leaf.
  if (!graphContainsClusters) {
    return kind === "knowledge";
  }
  if (tier === "knowledge") {
    return true;
  }
  return kind === "cluster";
}

export function shouldOfferDeeperNavigation(
  tier: SemanticZoomTier,
  containsClusters: boolean,
): boolean {
  return containsClusters && tier === "knowledge";
}

export function semanticZoomDescription(tier: SemanticZoomTier): string {
  switch (tier) {
    case "domains":
      return "Vue des grands domaines";
    case "themes":
      return "Vue des thèmes";
    case "knowledge":
      return "Vue des connaissances";
  }
}

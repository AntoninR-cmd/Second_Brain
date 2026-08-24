import { dimmedNodeColor, visualEdgeColor } from "./colors";
import { nodeKindVisibleAtZoom } from "./semanticZoom";
import type {
  BrainEdgePresentation,
  BrainGraph,
  BrainInteractionAction,
  BrainInteractionState,
  BrainNodePresentation,
  SemanticZoomTier,
} from "./types";

export const initialBrainInteractionState: BrainInteractionState = {
  currentClusterId: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  highlightedElementId: null,
  searchQuery: "",
  zoomTier: "domains",
  panelOpen: false,
};

export function brainInteractionReducer(
  state: BrainInteractionState,
  action: BrainInteractionAction,
): BrainInteractionState {
  switch (action.type) {
    case "enter-cluster":
      return {
        ...state,
        currentClusterId: action.clusterId,
        selectedNodeId: null,
        hoveredNodeId: null,
        highlightedElementId: `cluster:${action.clusterId}`,
        zoomTier: "themes",
        panelOpen: false,
      };
    case "go-root":
      return {
        ...state,
        currentClusterId: null,
        selectedNodeId: null,
        hoveredNodeId: null,
        highlightedElementId: null,
        zoomTier: "domains",
        panelOpen: false,
      };
    case "select-node":
      return {
        ...state,
        selectedNodeId: action.nodeId,
        highlightedElementId: action.nodeId,
        panelOpen: action.nodeId !== null,
      };
    case "hover-node":
      return { ...state, hoveredNodeId: action.nodeId };
    case "highlight-element":
      return { ...state, highlightedElementId: action.elementId };
    case "set-search":
      return { ...state, searchQuery: action.query };
    case "set-zoom-tier":
      return { ...state, zoomTier: action.tier };
    case "close-panel":
      return {
        ...state,
        selectedNodeId: null,
        panelOpen: false,
      };
  }
}

export interface BrainNeighborhood {
  focusNodeId: string | null;
  nodeIds: ReadonlySet<string>;
  edgeIds: ReadonlySet<string>;
}

export function directNeighborhood(
  graph: BrainGraph,
  state: Pick<BrainInteractionState, "selectedNodeId" | "hoveredNodeId">,
): BrainNeighborhood {
  // Selection is persistent and deliberately takes precedence over transient hover.
  const focusNodeId = state.selectedNodeId ?? state.hoveredNodeId;
  if (focusNodeId === null || !graph.hasNode(focusNodeId)) {
    return { focusNodeId: null, nodeIds: new Set(), edgeIds: new Set() };
  }

  const nodeIds = new Set<string>([focusNodeId, ...graph.neighbors(focusNodeId)]);
  const edgeIds = new Set<string>(graph.edges(focusNodeId));
  return { focusNodeId, nodeIds, edgeIds };
}

export function visibleLabelNodeIds(
  graph: BrainGraph,
  state: Pick<
    BrainInteractionState,
    "selectedNodeId" | "hoveredNodeId" | "highlightedElementId"
  >,
  tier: SemanticZoomTier,
): ReadonlySet<string> {
  const graphContainsClusters = graph.someNode(
    (_nodeId, attributes) => attributes.kind === "cluster",
  );
  // A leaf response contains only KnowledgeNodes. Its labels must not depend on
  // the camera event that usually promotes the React zoom tier after loading.
  const effectiveTier = graphContainsClusters ? tier : "knowledge";
  const limit =
    effectiveTier === "domains" ? 12 : effectiveTier === "themes" ? 18 : 24;
  const wantedKind = effectiveTier === "knowledge" ? "knowledge" : "cluster";
  const ordered = graph
    .filterNodes((_key, attributes) => attributes.kind === wantedKind)
    .sort((left, right) => {
      const leftAttributes = graph.getNodeAttributes(left);
      const rightAttributes = graph.getNodeAttributes(right);
      return (
        rightAttributes.rawSize - leftAttributes.rawSize ||
        leftAttributes.label.localeCompare(rightAttributes.label, "fr") ||
        left.localeCompare(right)
      );
    })
    .slice(0, limit);
  const visible = new Set(ordered);
  for (const nodeId of [
    state.selectedNodeId,
    state.hoveredNodeId,
    state.highlightedElementId,
  ]) {
    if (nodeId !== null && graph.hasNode(nodeId)) {
      visible.add(nodeId);
    }
  }
  return visible;
}

export interface BrainPresentationSnapshot {
  nodes: ReadonlyMap<string, BrainNodePresentation>;
  edges: ReadonlyMap<string, BrainEdgePresentation>;
  neighborhood: BrainNeighborhood;
}

export function buildPresentationSnapshot(
  graph: BrainGraph,
  state: Pick<
    BrainInteractionState,
    | "selectedNodeId"
    | "hoveredNodeId"
    | "highlightedElementId"
    | "zoomTier"
  >,
): BrainPresentationSnapshot {
  const neighborhood = directNeighborhood(graph, state);
  const labels = visibleLabelNodeIds(graph, state, state.zoomTier);
  const graphContainsClusters = graph.someNode(
    (_key, attributes) => attributes.kind === "cluster",
  );
  const scoreCutoff = defaultEdgeScoreCutoff(graph, state.zoomTier);
  const nodes = new Map<string, BrainNodePresentation>();
  const edges = new Map<string, BrainEdgePresentation>();

  graph.forEachNode((nodeId, attributes) => {
    const directlyRelated =
      neighborhood.focusNodeId === null || neighborhood.nodeIds.has(nodeId);
    const highlighted =
      nodeId === neighborhood.focusNodeId ||
      nodeId === state.highlightedElementId;
    const hidden =
      !highlighted &&
      !nodeKindVisibleAtZoom(
        attributes.kind,
        state.zoomTier,
        graphContainsClusters,
      );
    const dimmed = neighborhood.focusNodeId !== null && !directlyRelated;
    nodes.set(nodeId, {
      hidden,
      dimmed,
      highlighted,
      forceLabel: labels.has(nodeId),
      color: dimmed ? dimmedNodeColor() : attributes.color,
      size: attributes.size * (highlighted ? 1.28 : 1),
      zIndex: highlighted ? 10 : attributes.zIndex,
    });
  });

  graph.forEachEdge((edgeId, attributes) => {
    const isDirect = neighborhood.edgeIds.has(edgeId);
    const hasFocus = neighborhood.focusNodeId !== null;
    const hidden = hasFocus ? !isDirect : attributes.score < scoreCutoff;
    edges.set(edgeId, {
      hidden,
      dimmed: hasFocus && !isDirect,
      highlighted: isDirect,
      color: visualEdgeColor(attributes.score, isDirect),
      size: attributes.size * (isDirect ? 1.9 : 1),
      zIndex: isDirect ? 8 : attributes.zIndex,
    });
  });

  return { nodes, edges, neighborhood };
}

export function defaultEdgeScoreCutoff(
  graph: BrainGraph,
  tier: SemanticZoomTier,
): number {
  const scores = graph
    .mapEdges((_edge, attributes) => attributes.score)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (scores.length === 0 || tier === "domains") {
    return Number.NEGATIVE_INFINITY;
  }
  // Relative quantiles avoid imposing a corpus-specific semantic threshold.
  return quantile(scores, tier === "themes" ? 0.15 : 0.25);
}

function quantile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) {
    return Number.NEGATIVE_INFINITY;
  }
  const index = Math.floor((sorted.length - 1) * fraction);
  return sorted[index] ?? sorted[0] ?? Number.NEGATIVE_INFINITY;
}

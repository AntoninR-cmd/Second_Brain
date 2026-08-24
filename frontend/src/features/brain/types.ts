import type { UndirectedGraph } from "graphology";

import type { BrainGraphNodeKind } from "./contracts";

export interface BrainNodeAttributes {
  x: number;
  y: number;
  size: number;
  rawSize: number;
  label: string;
  color: string;
  kind: BrainGraphNodeKind;
  clusterId: string | null;
  knowledgeNodeId: string | null;
  sourceId: string | null;
  tags: string[];
  href: string | null;
  domainFamilyId: string;
  hidden: boolean;
  highlighted: boolean;
  forceLabel: boolean;
  zIndex: number;
}

export interface BrainEdgeAttributes {
  size: number;
  color: string;
  score: number;
  relationCount: number;
  hidden: boolean;
  highlighted: boolean;
  zIndex: number;
}

export interface BrainGraphAttributes {
  profileId: string;
  level: number;
  parentClusterId: string | null;
  truncated: boolean;
}

export type BrainGraph = UndirectedGraph<
  BrainNodeAttributes,
  BrainEdgeAttributes,
  BrainGraphAttributes
>;

export type SemanticZoomTier = "domains" | "themes" | "knowledge";

export interface BrainInteractionState {
  currentClusterId: string | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  highlightedElementId: string | null;
  searchQuery: string;
  zoomTier: SemanticZoomTier;
  panelOpen: boolean;
}

export type BrainInteractionAction =
  | { type: "enter-cluster"; clusterId: string }
  | { type: "go-root" }
  | { type: "select-node"; nodeId: string | null }
  | { type: "hover-node"; nodeId: string | null }
  | { type: "highlight-element"; elementId: string | null }
  | { type: "set-search"; query: string }
  | { type: "set-zoom-tier"; tier: SemanticZoomTier }
  | { type: "close-panel" };

export interface BrainNodePresentation {
  hidden: boolean;
  dimmed: boolean;
  highlighted: boolean;
  forceLabel: boolean;
  color: string;
  size: number;
  zIndex: number;
}

export interface BrainEdgePresentation {
  hidden: boolean;
  dimmed: boolean;
  highlighted: boolean;
  color: string;
  size: number;
  zIndex: number;
}

import { describe, expect, it } from "vitest";

import {
  brainInteractionReducer,
  buildBrainGraph,
  buildPresentationSnapshot,
  defaultEdgeScoreCutoff,
  directNeighborhood,
  initialBrainInteractionState,
  nodeKindVisibleAtZoom,
  semanticZoomTierForRatio,
  shouldOfferDeeperNavigation,
  visibleLabelNodeIds,
} from "../index";
import type { BrainGraphApiNode, BrainGraphApiResponse } from "../index";

function knowledgeNode(index: number): BrainGraphApiNode {
  return {
    id: `knowledge:k${index}`,
    kind: "knowledge",
    label: `Connaissance ${index.toString().padStart(2, "0")}`,
    x: index / 20,
    y: -index / 30,
    size: 1,
    cluster_id: "theme-1",
    knowledge_node_id: `k${index}`,
    source_id: "source-1",
    tags: index === 1 ? ["volume"] : [],
    href: `/connaissances/k${index}`,
  };
}

function knowledgeGraph(count = 4): ReturnType<typeof buildBrainGraph> {
  const nodes = Array.from({ length: count }, (_, index) => knowledgeNode(index + 1));
  const edges = Array.from({ length: Math.max(0, count - 1) }, (_, index) => ({
    source: `knowledge:k${index + 1}`,
    target: `knowledge:k${index + 2}`,
    score: 0.1 + index * 0.1,
    relation_count: 1,
  }));
  const payload: BrainGraphApiResponse = {
    profile_id: "profile-1",
    level: 3,
    parent_cluster_id: "theme-1",
    nodes,
    edges,
    truncated: false,
  };
  return buildBrainGraph(payload, { domainFamilyId: "domain-1" });
}

describe("semantic zoom", () => {
  it("convertit le ratio caméra Sigma en trois niveaux stables", () => {
    expect(semanticZoomTierForRatio(1.8)).toBe("domains");
    expect(semanticZoomTierForRatio(0.8)).toBe("themes");
    expect(semanticZoomTierForRatio(0.3)).toBe("knowledge");
  });

  it("n'efface jamais un payload feuille composé uniquement de connaissances", () => {
    expect(nodeKindVisibleAtZoom("knowledge", "domains", false)).toBe(true);
    expect(nodeKindVisibleAtZoom("knowledge", "domains", true)).toBe(false);
    expect(nodeKindVisibleAtZoom("cluster", "themes", true)).toBe(true);
  });

  it("propose l'entrée dans un cluster seulement au zoom rapproché", () => {
    expect(shouldOfferDeeperNavigation("knowledge", true)).toBe(true);
    expect(shouldOfferDeeperNavigation("themes", true)).toBe(false);
    expect(shouldOfferDeeperNavigation("knowledge", false)).toBe(false);
  });
});

describe("interactions et visibilité", () => {
  it("conserve dans React l'état de navigation, de sélection et du panneau", () => {
    const entered = brainInteractionReducer(initialBrainInteractionState, {
      type: "enter-cluster",
      clusterId: "domain-1",
    });
    const selected = brainInteractionReducer(entered, {
      type: "select-node",
      nodeId: "knowledge:k1",
    });

    expect(entered.currentClusterId).toBe("domain-1");
    expect(entered.zoomTier).toBe("themes");
    expect(selected.panelOpen).toBe(true);
    expect(selected.selectedNodeId).toBe("knowledge:k1");

    const closed = brainInteractionReducer(selected, { type: "close-panel" });
    expect(closed.panelOpen).toBe(false);
    expect(closed.selectedNodeId).toBeNull();
  });

  it("donne priorité à la sélection persistante sur le hover", () => {
    const graph = knowledgeGraph();
    const neighborhood = directNeighborhood(graph, {
      selectedNodeId: "knowledge:k1",
      hoveredNodeId: "knowledge:k3",
    });

    expect(neighborhood.focusNodeId).toBe("knowledge:k1");
    expect(neighborhood.nodeIds).toEqual(
      new Set(["knowledge:k1", "knowledge:k2"]),
    );
    expect(neighborhood.edgeIds.size).toBe(1);
  });

  it("met en évidence le nœud, ses voisins et ses seules arêtes directes", () => {
    const graph = knowledgeGraph();
    const snapshot = buildPresentationSnapshot(graph, {
      selectedNodeId: "knowledge:k2",
      hoveredNodeId: null,
      highlightedElementId: null,
      zoomTier: "knowledge",
    });

    expect(snapshot.nodes.get("knowledge:k2")?.highlighted).toBe(true);
    expect(snapshot.nodes.get("knowledge:k1")?.dimmed).toBe(false);
    expect(snapshot.nodes.get("knowledge:k4")?.dimmed).toBe(true);
    const directEdgeIds = snapshot.neighborhood.edgeIds;
    for (const [edgeId, presentation] of snapshot.edges) {
      expect(presentation.hidden).toBe(!directEdgeIds.has(edgeId));
      expect(presentation.highlighted).toBe(directEdgeIds.has(edgeId));
    }
  });

  it("limite les labels KnowledgeNode et conserve hover/sélection", () => {
    const graph = knowledgeGraph(30);
    const labels = visibleLabelNodeIds(
      graph,
      {
        selectedNodeId: "knowledge:k30",
        hoveredNodeId: null,
        highlightedElementId: null,
      },
      "knowledge",
    );

    expect(labels.size).toBe(25);
    expect(labels.has("knowledge:k30")).toBe(true);
  });

  it("affiche les labels d'une feuille sans attendre un evenement camera", () => {
    const graph = knowledgeGraph(4);
    const labels = visibleLabelNodeIds(
      graph,
      {
        selectedNodeId: null,
        hoveredNodeId: null,
        highlightedElementId: null,
      },
      "themes",
    );

    expect(labels).toEqual(
      new Set([
        "knowledge:k1",
        "knowledge:k2",
        "knowledge:k3",
        "knowledge:k4",
      ]),
    );
  });

  it("masque relativement les arêtes les plus faibles sans seuil sémantique fixe", () => {
    const graph = knowledgeGraph(6);
    const cutoff = defaultEdgeScoreCutoff(graph, "knowledge");
    const snapshot = buildPresentationSnapshot(graph, {
      selectedNodeId: null,
      hoveredNodeId: null,
      highlightedElementId: null,
      zoomTier: "knowledge",
    });

    expect(cutoff).toBeCloseTo(0.2);
    expect([...snapshot.edges.values()].filter((edge) => edge.hidden)).toHaveLength(1);
  });
});

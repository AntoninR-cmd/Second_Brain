import { describe, expect, it } from "vitest";

import {
  BRAIN_ROOT_BREADCRUMB,
  breadcrumbTarget,
  buildBrainBreadcrumb,
  firstDomainId,
  mergeSearchItems,
  navigationTargetForSearchResult,
  normalizeBrainSearchText,
  searchBrainItems,
  searchItemsFromClusters,
  searchItemsFromKnowledge,
} from "../index";
import type {
  BrainClusterApiItem,
  BrainKnowledgeApiItem,
  BrainSearchItem,
} from "../index";

const clusters: BrainClusterApiItem[] = [
  {
    id: "root",
    parent_id: null,
    level: 0,
    label: "Second Brain",
    description: null,
    label_source: "deterministic",
    member_count: 66,
    representative_knowledge_node_ids: [],
    x: 0,
    y: 0,
    child_count: 2,
  },
  {
    id: "training",
    parent_id: "root",
    level: 1,
    label: "Principes d'entraînement en musculation",
    description: null,
    label_source: "ollama",
    member_count: 39,
    representative_knowledge_node_ids: [],
    x: -0.5,
    y: 0.2,
    child_count: 4,
  },
  {
    id: "periodization",
    parent_id: "training",
    level: 2,
    label: "Programmation périodisée en sport",
    description: null,
    label_source: "ollama",
    member_count: 5,
    representative_knowledge_node_ids: [],
    x: -0.2,
    y: 0.4,
    child_count: 0,
  },
];

const knowledge: BrainKnowledgeApiItem[] = [
  {
    id: "k1",
    cluster_id: "periodization",
    title: "Durée typique d'un mésocycle",
    tags: ["entraînement", "périodisation"],
    source_id: "source-1",
    source_title: "Programmation sportive",
    x: -0.15,
    y: 0.42,
    is_unassigned: false,
    href: "/connaissances/k1",
  },
];

describe("recherche locale du cerveau", () => {
  it("normalise accents et apostrophes", () => {
    expect(normalizeBrainSearchText("Périodisation d’entraînement")).toBe(
      "periodisation d entrainement",
    );
  });

  it("retrouve clusters par nom et connaissances par titre", () => {
    const items = mergeSearchItems(
      searchItemsFromClusters(clusters),
      searchItemsFromKnowledge(knowledge),
    );

    expect(searchBrainItems("musculation", items)[0]?.id).toBe("training");
    expect(searchBrainItems("mesocycle", items)[0]?.id).toBe("k1");
  });

  it("retrouve une connaissance par tag sans dupliquer les entrées", () => {
    const nodeItems = searchItemsFromKnowledge(knowledge);
    const items = mergeSearchItems(nodeItems, nodeItems);
    const results = searchBrainItems("periodisation", items);

    expect(items).toHaveLength(1);
    expect(results).toHaveLength(1);
    expect(results[0]?.kind).toBe("knowledge");
  });

  it("classe une correspondance exacte avant une correspondance partielle", () => {
    const items: BrainSearchItem[] = [
      {
        id: "partial",
        nodeKey: "cluster:partial",
        kind: "cluster",
        label: "Gestion du volume",
        clusterId: "partial",
        href: null,
        tags: [],
        subtitle: null,
      },
      {
        id: "exact",
        nodeKey: "knowledge:exact",
        kind: "knowledge",
        label: "Volume",
        clusterId: "training",
        href: "/connaissances/exact",
        tags: [],
        subtitle: null,
      },
    ];

    expect(searchBrainItems("volume", items).map((item) => item.id)).toEqual([
      "exact",
      "partial",
    ]);
  });

  it("ne retourne rien pour une saisie vide", () => {
    expect(searchBrainItems("   ", searchItemsFromClusters(clusters))).toEqual([]);
  });
});

describe("breadcrumb et navigation", () => {
  it("construit le chemin Second Brain vers le thème", () => {
    const breadcrumb = buildBrainBreadcrumb("periodization", clusters);

    expect(breadcrumb.map((item) => item.label)).toEqual([
      "Second Brain",
      "Principes d'entraînement en musculation",
      "Programmation périodisée en sport",
    ]);
    expect(firstDomainId(breadcrumb)).toBe("training");
  });

  it("revient proprement à la racine pour un cluster inconnu", () => {
    expect(buildBrainBreadcrumb("missing", clusters)).toEqual([
      BRAIN_ROOT_BREADCRUMB,
    ]);
  });

  it("supporte les cibles intermédiaires du fil d'Ariane", () => {
    const breadcrumb = buildBrainBreadcrumb("periodization", clusters);

    expect(breadcrumbTarget(breadcrumb, 0)?.clusterId).toBeNull();
    expect(breadcrumbTarget(breadcrumb, 1)?.clusterId).toBe("training");
    expect(breadcrumbTarget(breadcrumb, 99)).toBeNull();
  });

  it("transforme un résultat KnowledgeNode en cible de caméra et de panneau", () => {
    const result = searchBrainItems(
      "mesocycle",
      searchItemsFromKnowledge(knowledge),
    )[0];
    if (result === undefined) {
      throw new Error("résultat de test absent");
    }

    expect(navigationTargetForSearchResult(result)).toEqual({
      clusterId: "periodization",
      graphNodeId: "knowledge:k1",
      knowledgeNodeId: "k1",
      href: "/connaissances/k1",
    });
  });
});

import { describe, expect, it } from "vitest";

import {
  BrainGraphContractError,
  brainFamilyHue,
  buildBrainGraph,
  colorForBrainNode,
  edgeKey,
  visualNodeSize,
} from "../index";
import type { BrainGraphApiResponse } from "../index";

function graphPayload(): BrainGraphApiResponse {
  return {
    profile_id: "profile-1",
    level: 1,
    parent_cluster_id: null,
    truncated: false,
    nodes: [
      {
        id: "cluster:c1",
        kind: "cluster",
        label: "Musculation",
        x: -0.75,
        y: 0.2,
        size: 39,
        cluster_id: "c1",
        knowledge_node_id: null,
        source_id: null,
        tags: [],
        href: null,
      },
      {
        id: "cluster:c2",
        kind: "cluster",
        label: "Style vestimentaire",
        x: 0.7,
        y: -0.1,
        size: 27,
        cluster_id: "c2",
        knowledge_node_id: null,
        source_id: null,
        tags: [],
        href: null,
      },
    ],
    edges: [
      {
        source: "cluster:c1",
        target: "cluster:c2",
        score: 0.62,
        relation_count: 12,
      },
    ],
  };
}

describe("buildBrainGraph", () => {
  it("conserve les coordonnées Phase 6A et les métadonnées du profil", () => {
    const graph = buildBrainGraph(graphPayload());

    expect(graph.order).toBe(2);
    expect(graph.size).toBe(1);
    expect(graph.getAttribute("profileId")).toBe("profile-1");
    expect(graph.getAttribute("level")).toBe(1);
    expect(graph.getNodeAttribute("cluster:c1", "x")).toBe(-0.75);
    expect(graph.getNodeAttribute("cluster:c1", "y")).toBe(0.2);
    expect(graph.getNodeAttribute("cluster:c1", "rawSize")).toBe(39);
  });

  it("transforme une relation en arête Graphology discrète sans modifier son score", () => {
    const graph = buildBrainGraph(graphPayload());
    const key = edgeKey("cluster:c2", "cluster:c1");

    expect(graph.hasEdge(key)).toBe(true);
    expect(graph.getEdgeAttribute(key, "score")).toBe(0.62);
    expect(graph.getEdgeAttribute(key, "relationCount")).toBe(12);
    expect(graph.getEdgeAttribute(key, "size")).toBeLessThanOrEqual(2.1);
    expect(graph.getEdgeAttribute(key, "color")).toMatch(/^rgba\(/);
  });

  it("déduplique défensivement les arêtes inversées", () => {
    const payload = graphPayload();
    payload.edges.push({
      source: "cluster:c2",
      target: "cluster:c1",
      score: 0.68,
      relation_count: 10,
    });

    const graph = buildBrainGraph(payload);

    expect(graph.size).toBe(1);
    expect(graph.getEdgeAttribute(edgeKey("cluster:c1", "cluster:c2"), "score")).toBe(
      0.68,
    );
    expect(
      graph.getEdgeAttribute(edgeKey("cluster:c1", "cluster:c2"), "relationCount"),
    ).toBe(12);
  });

  it("ignore les boucles et les arêtes dont une extrémité manque", () => {
    const payload = graphPayload();
    payload.edges = [
      { source: "cluster:c1", target: "cluster:c1", score: 1, relation_count: 1 },
      { source: "cluster:c1", target: "missing", score: 0.9, relation_count: 1 },
    ];

    expect(buildBrainGraph(payload).size).toBe(0);
  });

  it("rejette un identifiant de nœud dupliqué", () => {
    const payload = graphPayload();
    const first = payload.nodes[0];
    if (first === undefined) {
      throw new Error("fixture incomplète");
    }
    payload.nodes.push({ ...first });

    expect(() => buildBrainGraph(payload)).toThrow(BrainGraphContractError);
  });

  it("rejette des coordonnées qui ne peuvent pas être rendues", () => {
    const payload = graphPayload();
    const first = payload.nodes[0];
    if (first === undefined) {
      throw new Error("fixture incomplète");
    }
    first.x = Number.NaN;

    expect(() => buildBrainGraph(payload)).toThrow(/coordonnées 2D/);
  });

  it("borne les tailles visuelles sans amplifier arbitrairement les connaissances", () => {
    expect(visualNodeSize("knowledge", 1)).toBe(5.5);
    expect(visualNodeSize("knowledge", 100)).toBe(5.5);
    expect(visualNodeSize("cluster", 1)).toBeGreaterThan(visualNodeSize("knowledge", 1));
    expect(visualNodeSize("cluster", 1_000_000)).toBe(26);
  });
});

describe("couleurs déterministes", () => {
  it("retourne toujours la même couleur pour une même famille", () => {
    expect(colorForBrainNode("domain-1", "theme-1", "cluster")).toBe(
      colorForBrainNode("domain-1", "theme-1", "cluster"),
    );
    expect(brainFamilyHue("domain-1")).toBe(brainFamilyHue("domain-1"));
  });

  it("propage la famille de domaine fournie aux sous-thèmes", () => {
    const graph = buildBrainGraph(graphPayload(), { domainFamilyId: "domain-musculation" });

    expect(graph.getNodeAttribute("cluster:c1", "domainFamilyId")).toBe(
      "domain-musculation",
    );
    expect(graph.getNodeAttribute("cluster:c2", "domainFamilyId")).toBe(
      "domain-musculation",
    );
  });
});

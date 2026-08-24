import { UndirectedGraph } from "graphology";

import {
  colorForBrainNode,
  visualEdgeColor,
  visualEdgeSize,
  visualNodeSize,
} from "./colors";
import type {
  BrainGraphApiEdge,
  BrainGraphApiNode,
  BrainGraphApiResponse,
} from "./contracts";
import type {
  BrainEdgeAttributes,
  BrainGraph,
  BrainGraphAttributes,
  BrainNodeAttributes,
} from "./types";

export interface BuildBrainGraphOptions {
  /** The top-level domain currently open, used to preserve its color family. */
  domainFamilyId?: string | null;
  /** Optional ancestry map when nodes from several domains share one payload. */
  domainFamilyByClusterId?: Readonly<Record<string, string>>;
}

export class BrainGraphContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BrainGraphContractError";
  }
}

export function buildBrainGraph(
  payload: BrainGraphApiResponse,
  options: BuildBrainGraphOptions = {},
): BrainGraph {
  const graph = new UndirectedGraph<
    BrainNodeAttributes,
    BrainEdgeAttributes,
    BrainGraphAttributes
  >({ allowSelfLoops: false });

  graph.replaceAttributes({
    profileId: payload.profile_id,
    level: payload.level,
    parentClusterId: payload.parent_cluster_id,
    truncated: payload.truncated,
  });

  for (const node of payload.nodes) {
    assertGraphNode(node);
    if (graph.hasNode(node.id)) {
      throw new BrainGraphContractError(
        `Le nœud ${node.id} est présent plusieurs fois dans la réponse BrainGraph.`,
      );
    }
    const familyId = resolveDomainFamily(node, options);
    graph.addNode(node.id, {
      x: node.x,
      y: node.y,
      size: visualNodeSize(node.kind, node.size),
      rawSize: node.size,
      label: node.label,
      color: colorForBrainNode(familyId, node.cluster_id ?? node.id, node.kind),
      kind: node.kind,
      clusterId: node.cluster_id,
      knowledgeNodeId: node.knowledge_node_id,
      sourceId: node.source_id,
      tags: [...node.tags],
      href: node.href,
      domainFamilyId: familyId,
      hidden: false,
      highlighted: false,
      forceLabel: false,
      zIndex: node.kind === "cluster" ? 2 : 1,
    });
  }

  const canonicalEdges = canonicalizeEdges(payload.edges, graph);
  for (const edge of canonicalEdges.values()) {
    graph.addUndirectedEdgeWithKey(edgeKey(edge.source, edge.target), edge.source, edge.target, {
      size: visualEdgeSize(edge.relation_count),
      color: visualEdgeColor(edge.score),
      score: edge.score,
      relationCount: edge.relation_count,
      hidden: false,
      highlighted: false,
      zIndex: 0,
    });
  }

  return graph;
}

export function edgeKey(source: string, target: string): string {
  const [left, right] = canonicalPair(source, target);
  return `edge:${encodeURIComponent(left)}::${encodeURIComponent(right)}`;
}

function canonicalizeEdges(
  edges: readonly BrainGraphApiEdge[],
  graph: BrainGraph,
): Map<string, BrainGraphApiEdge> {
  const result = new Map<string, BrainGraphApiEdge>();
  for (const edge of edges) {
    if (
      edge.source === edge.target ||
      !graph.hasNode(edge.source) ||
      !graph.hasNode(edge.target)
    ) {
      continue;
    }
    const [source, target] = canonicalPair(edge.source, edge.target);
    const key = edgeKey(source, target);
    const previous = result.get(key);
    if (previous === undefined) {
      result.set(key, {
        source,
        target,
        score: clampScore(edge.score),
        relation_count: Math.max(1, Math.trunc(edge.relation_count)),
      });
      continue;
    }
    // The API normally guarantees uniqueness. Keeping the strongest observation
    // makes the renderer defensive without recomputing a semantic score.
    result.set(key, {
      source,
      target,
      score: Math.max(previous.score, clampScore(edge.score)),
      relation_count: Math.max(
        previous.relation_count,
        Math.max(1, Math.trunc(edge.relation_count)),
      ),
    });
  }
  return result;
}

function resolveDomainFamily(
  node: BrainGraphApiNode,
  options: BuildBrainGraphOptions,
): string {
  const mapped =
    node.cluster_id === null
      ? undefined
      : options.domainFamilyByClusterId?.[node.cluster_id];
  return (
    mapped ??
    options.domainFamilyId ??
    (node.kind === "cluster" ? node.cluster_id : null) ??
    "unassigned"
  );
}

function assertGraphNode(node: BrainGraphApiNode): void {
  if (node.id.length === 0 || node.label.trim().length === 0) {
    throw new BrainGraphContractError("Un nœud BrainGraph ne possède pas d'identité lisible.");
  }
  if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
    throw new BrainGraphContractError(
      `Le nœud ${node.id} ne possède pas de coordonnées 2D valides.`,
    );
  }
}

function canonicalPair(source: string, target: string): [string, string] {
  return source.localeCompare(target) <= 0 ? [source, target] : [target, source];
}

function clampScore(score: number): number {
  if (!Number.isFinite(score)) {
    return 0;
  }
  return Math.min(1, Math.max(-1, score));
}

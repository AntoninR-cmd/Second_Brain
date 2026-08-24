import type {
  BrainClusterApiItem,
  BrainKnowledgeApiItem,
} from "./contracts";
import type { BrainGraph } from "./types";

export type BrainSearchResultKind = "cluster" | "knowledge";

export interface BrainSearchItem {
  id: string;
  nodeKey: string;
  kind: BrainSearchResultKind;
  label: string;
  clusterId: string | null;
  href: string | null;
  tags: string[];
  subtitle: string | null;
}

export interface BrainSearchResult extends BrainSearchItem {
  matchScore: number;
}

export function searchItemsFromGraph(graph: BrainGraph): BrainSearchItem[] {
  return graph.mapNodes((nodeKey, attributes) => ({
    id: attributes.knowledgeNodeId ?? attributes.clusterId ?? nodeKey,
    nodeKey,
    kind: attributes.kind,
    label: attributes.label,
    clusterId: attributes.clusterId,
    href: attributes.href,
    tags: [...attributes.tags],
    subtitle:
      attributes.kind === "cluster"
        ? `${attributes.rawSize} connaissance${attributes.rawSize > 1 ? "s" : ""}`
        : attributes.tags.length > 0
          ? attributes.tags.map((tag) => `#${tag}`).join(" ")
          : null,
  }));
}

export function searchItemsFromClusters(
  clusters: readonly BrainClusterApiItem[],
): BrainSearchItem[] {
  return clusters.map((cluster) => ({
    id: cluster.id,
    nodeKey: `cluster:${cluster.id}`,
    kind: "cluster",
    label: cluster.label,
    clusterId: cluster.id,
    href: null,
    tags: [],
    subtitle: `${cluster.member_count} connaissance${cluster.member_count > 1 ? "s" : ""}`,
  }));
}

export function searchItemsFromKnowledge(
  nodes: readonly BrainKnowledgeApiItem[],
): BrainSearchItem[] {
  return nodes.map((node) => ({
    id: node.id,
    nodeKey: `knowledge:${node.id}`,
    kind: "knowledge",
    label: node.title,
    clusterId: node.cluster_id,
    href: node.href,
    tags: [...node.tags],
    subtitle: node.source_title,
  }));
}

export function mergeSearchItems(
  ...collections: ReadonlyArray<readonly BrainSearchItem[]>
): BrainSearchItem[] {
  const unique = new Map<string, BrainSearchItem>();
  for (const collection of collections) {
    for (const item of collection) {
      unique.set(`${item.kind}:${item.id}`, item);
    }
  }
  return [...unique.values()];
}

export function searchBrainItems(
  query: string,
  items: readonly BrainSearchItem[],
  limit = 12,
): BrainSearchResult[] {
  const normalizedQuery = normalizeBrainSearchText(query.trim());
  if (normalizedQuery.length === 0 || limit <= 0) {
    return [];
  }

  return items
    .map((item): BrainSearchResult | null => {
      const label = normalizeBrainSearchText(item.label);
      const tags = item.tags.map(normalizeBrainSearchText);
      const matchScore = scoreMatch(normalizedQuery, label, tags);
      return matchScore === 0 ? null : { ...item, matchScore };
    })
    .filter((item): item is BrainSearchResult => item !== null)
    .sort(
      (left, right) =>
        right.matchScore - left.matchScore ||
        (left.kind === right.kind ? 0 : left.kind === "cluster" ? -1 : 1) ||
        left.label.localeCompare(right.label, "fr") ||
        left.id.localeCompare(right.id),
    )
    .slice(0, limit);
}

export function normalizeBrainSearchText(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("fr")
    .replace(/[’']/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function scoreMatch(query: string, label: string, tags: readonly string[]): number {
  if (label === query) {
    return 100;
  }
  if (label.startsWith(query)) {
    return 85;
  }
  if (label.split(" ").some((word) => word.startsWith(query))) {
    return 72;
  }
  if (label.includes(query)) {
    return 60;
  }
  if (tags.some((tag) => tag === query)) {
    return 48;
  }
  if (tags.some((tag) => tag.includes(query))) {
    return 36;
  }
  return 0;
}

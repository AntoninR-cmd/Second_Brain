import type { BrainClusterApiItem } from "./contracts";
import type { BrainSearchResult } from "./search";

export interface BrainBreadcrumbItem {
  clusterId: string | null;
  label: string;
  level: number;
}

export const BRAIN_ROOT_BREADCRUMB: BrainBreadcrumbItem = Object.freeze({
  clusterId: null,
  label: "Second Brain",
  level: 0,
});

export function buildBrainBreadcrumb(
  currentClusterId: string | null,
  clusters: readonly BrainClusterApiItem[],
): BrainBreadcrumbItem[] {
  if (currentClusterId === null) {
    return [BRAIN_ROOT_BREADCRUMB];
  }
  const clustersById = new Map(clusters.map((cluster) => [cluster.id, cluster]));
  const reversedPath: BrainClusterApiItem[] = [];
  const visited = new Set<string>();
  let current = clustersById.get(currentClusterId);
  while (current !== undefined && !visited.has(current.id)) {
    visited.add(current.id);
    if (current.level > 0) {
      reversedPath.push(current);
    }
    current =
      current.parent_id === null ? undefined : clustersById.get(current.parent_id);
  }
  reversedPath.reverse();
  return [
    BRAIN_ROOT_BREADCRUMB,
    ...reversedPath.map((cluster) => ({
      clusterId: cluster.id,
      label: cluster.label,
      level: cluster.level,
    })),
  ];
}

export function breadcrumbTarget(
  breadcrumb: readonly BrainBreadcrumbItem[],
  index: number,
): BrainBreadcrumbItem | null {
  return index >= 0 && index < breadcrumb.length
    ? (breadcrumb[index] ?? null)
    : null;
}

export interface BrainNavigationTarget {
  clusterId: string | null;
  graphNodeId: string;
  knowledgeNodeId: string | null;
  href: string | null;
}

export function navigationTargetForSearchResult(
  result: BrainSearchResult,
): BrainNavigationTarget {
  return {
    clusterId: result.clusterId,
    graphNodeId: result.nodeKey,
    knowledgeNodeId: result.kind === "knowledge" ? result.id : null,
    href: result.href,
  };
}

export function firstDomainId(
  breadcrumb: readonly BrainBreadcrumbItem[],
): string | null {
  return breadcrumb.find((item) => item.level === 1)?.clusterId ?? null;
}

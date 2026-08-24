import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import { Link } from "react-router-dom";

import {
  getBrainCluster,
  getBrainClusters,
  getBrainGraph,
  getBrainStatus,
  getKnowledgeNode,
  getReadableError,
  rebuildBrain,
  searchBrain,
} from "../api/client";
import type {
  BrainCluster,
  BrainSearchResult,
  BrainStatus,
} from "../api/types";
import {
  BrainCanvas,
  type BrainCanvasHandle,
  type BrainCanvasMetric,
} from "../features/brain/BrainCanvas";
import {
  buildBrainGraph,
  buildPresentationSnapshot,
  brainInteractionReducer,
  initialBrainInteractionState,
  semanticZoomDescription,
  semanticZoomTierForRatio,
  shouldOfferDeeperNavigation,
  type BrainGraph,
} from "../features/brain";
import {
  BrainBreadcrumb,
  type BrainBreadcrumbItem,
} from "../features/brain/BrainBreadcrumb";
import { BrainDetailsPanel } from "../features/brain/BrainDetailsPanel";
import { BrainSearch } from "../features/brain/BrainSearch";
import {
  BrainLegend,
  type BrainPerformanceMetrics,
  BrainToolbar,
} from "../features/brain/BrainToolbar";
import { getBrainAvailability } from "../features/brain/status";

const ROOT_LEVEL = 1;
const AUTO_ENTER_RATIO = 0.52;
const AUTO_EXIT_RATIO = 1.85;
const SEARCH_DEBOUNCE_MS = 220;

interface SelectedBrainTarget {
  kind: "cluster" | "knowledge";
  id: string;
  graphNodeId: string;
}

interface TimedGraphResponse {
  data: Awaited<ReturnType<typeof getBrainGraph>>;
  durationMs: number;
}

interface GraphBuildResult {
  graph: BrainGraph | null;
  durationMs: number | null;
  error: string | null;
}

function clockNow(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timeout = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timeout);
  }, [delay, value]);
  return debounced;
}

function rootBreadcrumb(cluster: BrainCluster | undefined): BrainBreadcrumbItem {
  return {
    id: cluster?.id ?? "second-brain-root",
    label: cluster?.label ?? "Second Brain",
    level: 0,
  };
}

function graphNodeId(kind: "cluster" | "knowledge", id: string): string {
  return `${kind}:${id}`;
}

function hasActiveJob(status: BrainStatus | undefined): boolean {
  return status?.active_job?.status === "pending" || status?.active_job?.status === "running";
}

export function BrainPage() {
  const queryClient = useQueryClient();
  const canvasRef = useRef<BrainCanvasHandle | null>(null);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const navigationLockRef = useRef(false);
  const previousCameraRatioRef = useRef(1);
  const pendingFocusRef = useRef<string | null>(null);
  const navigationStartedAtRef = useRef<number | null>(null);
  const renderedGraphRef = useRef<BrainGraph | null>(null);
  const [breadcrumbs, setBreadcrumbs] = useState<BrainBreadcrumbItem[]>([]);
  const [selection, setSelection] = useState<SelectedBrainTarget | null>(null);
  const [searchValue, setSearchValue] = useState("");
  const debouncedSearch = useDebouncedValue(searchValue.trim(), SEARCH_DEBOUNCE_MS);
  const [interaction, dispatch] = useReducer(
    brainInteractionReducer,
    initialBrainInteractionState,
  );
  const [metrics, setMetrics] = useState<BrainPerformanceMetrics>({
    apiMs: null,
    graphologyMs: null,
    firstPaintMs: null,
    navigationMs: null,
    hoverMs: null,
  });

  const statusQuery = useQuery({
    queryKey: ["brain", "status"],
    queryFn: getBrainStatus,
    refetchInterval: (query) => (hasActiveJob(query.state.data) ? 1_000 : 15_000),
  });
  const status = statusQuery.data;
  const availability = status ? getBrainAvailability(status) : null;
  const profileId = status?.active_profile?.id ?? null;

  const rootQuery = useQuery({
    queryKey: ["brain", "clusters", profileId, "root"],
    queryFn: () => getBrainClusters({ level: 0, limit: 10 }),
    enabled: Boolean(profileId && availability?.canDisplayGraph),
    staleTime: 60_000,
  });
  const rootItem = useMemo(
    () => rootBreadcrumb(rootQuery.data?.[0]),
    [rootQuery.data],
  );

  useEffect(() => {
    if (breadcrumbs.length === 0 && profileId) {
      setBreadcrumbs([rootItem]);
    }
  }, [breadcrumbs.length, profileId, rootItem]);

  useEffect(() => {
    setBreadcrumbs([]);
    setSelection(null);
    dispatch({ type: "go-root" });
  }, [profileId]);

  const currentBreadcrumb = breadcrumbs.at(-1) ?? rootItem;
  const currentClusterId =
    currentBreadcrumb.level > 0 ? currentBreadcrumb.id : null;
  const domainFamilyId =
    breadcrumbs.find((item) => item.level === 1)?.id ?? null;

  const graphQuery = useQuery({
    queryKey: ["brain", "graph", profileId, currentClusterId ?? "root"],
    queryFn: async (): Promise<TimedGraphResponse> => {
      const startedAt = clockNow();
      const data = await getBrainGraph(
        currentClusterId
          ? { clusterId: currentClusterId }
          : { level: ROOT_LEVEL },
      );
      return { data, durationMs: clockNow() - startedAt };
    },
    enabled: Boolean(profileId && availability?.canDisplayGraph),
    staleTime: 60_000,
  });

  const builtGraph = useMemo<GraphBuildResult>(() => {
    if (!graphQuery.data) {
      return { graph: null, durationMs: null, error: null };
    }
    const startedAt = clockNow();
    try {
      const graph = buildBrainGraph(graphQuery.data.data, { domainFamilyId });
      return {
        graph,
        durationMs: clockNow() - startedAt,
        error: null,
      };
    } catch (error) {
      return {
        graph: null,
        durationMs: clockNow() - startedAt,
        error: error instanceof Error ? error.message : "Le graphe reçu est invalide.",
      };
    }
  }, [domainFamilyId, graphQuery.data]);

  useEffect(() => {
    if (!graphQuery.data || builtGraph.durationMs === null) return;
    setMetrics((current) => ({
      ...current,
      apiMs: graphQuery.data.durationMs,
      graphologyMs: builtGraph.durationMs,
    }));
  }, [builtGraph.durationMs, graphQuery.data]);

  const graphContainsClusters =
    builtGraph.graph?.someNode((_node, attributes) => attributes.kind === "cluster") ??
    false;
  const presentation = useMemo(
    () =>
      builtGraph.graph
        ? buildPresentationSnapshot(builtGraph.graph, interaction)
        : null,
    [builtGraph.graph, interaction],
  );

  const clusterQuery = useQuery({
    queryKey: ["brain", "cluster", profileId, selection?.id],
    queryFn: () => getBrainCluster(selection?.id ?? ""),
    enabled: selection?.kind === "cluster",
    staleTime: 60_000,
  });
  const knowledgeQuery = useQuery({
    queryKey: ["knowledge-nodes", selection?.id],
    queryFn: () => getKnowledgeNode(selection?.id ?? ""),
    enabled: selection?.kind === "knowledge",
    staleTime: 60_000,
  });

  const searchQuery = useQuery({
    queryKey: ["brain", "search", profileId, debouncedSearch],
    queryFn: () => searchBrain(debouncedSearch, 20),
    enabled: Boolean(profileId && debouncedSearch.length >= 2),
    staleTime: 30_000,
  });

  const rebuildMutation = useMutation({
    mutationFn: rebuildBrain,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["brain", "status"] });
    },
  });

  const clearSelection = useCallback(() => {
    setSelection(null);
    dispatch({ type: "select-node", nodeId: null });
    dispatch({ type: "highlight-element", elementId: null });
  }, []);

  const enterCluster = useCallback(
    async (target: { id: string; label: string; level: number }) => {
      const graph = builtGraph.graph;
      const targetNodeId = graphNodeId("cluster", target.id);
      if (navigationLockRef.current) return;
      navigationLockRef.current = true;
      navigationStartedAtRef.current = clockNow();
      if (graph?.hasNode(targetNodeId)) {
        await canvasRef.current?.focusNode(targetNodeId, {
          ratio: 0.3,
          duration: 260,
        });
      }
      setBreadcrumbs((current) => {
        const existingIndex = current.findIndex((item) => item.id === target.id);
        if (existingIndex >= 0) return current.slice(0, existingIndex + 1);
        return [...(current.length > 0 ? current : [rootItem]), target];
      });
      clearSelection();
      dispatch({ type: "enter-cluster", clusterId: target.id });
    },
    [builtGraph.graph, clearSelection, rootItem],
  );

  const navigateBreadcrumb = useCallback(
    (index: number) => {
      if (index < 0 || index >= breadcrumbs.length || navigationLockRef.current) {
        return;
      }
      navigationLockRef.current = true;
      navigationStartedAtRef.current = clockNow();
      const next = breadcrumbs.slice(0, index + 1);
      setBreadcrumbs(next);
      clearSelection();
      const target = next.at(-1);
      if (!target || target.level === 0) {
        dispatch({ type: "go-root" });
      } else {
        dispatch({ type: "enter-cluster", clusterId: target.id });
      }
    },
    [breadcrumbs, clearSelection],
  );

  const selectGraphNode = useCallback(
    (nodeId: string) => {
      const graph = builtGraph.graph;
      if (!graph?.hasNode(nodeId)) return;
      const attributes = graph.getNodeAttributes(nodeId);
      const targetId =
        attributes.kind === "cluster"
          ? attributes.clusterId
          : attributes.knowledgeNodeId;
      if (!targetId) return;
      setSelection({ kind: attributes.kind, id: targetId, graphNodeId: nodeId });
      dispatch({ type: "select-node", nodeId });
      void canvasRef.current?.focusNode(nodeId, {
        ratio: Math.min(
          previousCameraRatioRef.current,
          attributes.kind === "cluster" ? 0.68 : 0.42,
        ),
        duration: 220,
      });
    },
    [builtGraph.graph],
  );

  const handleSearchSelection = useCallback(
    (result: BrainSearchResult) => {
      const nextBreadcrumbs = result.ancestors.map((ancestor) => ({
        id: ancestor.id,
        label: ancestor.label,
        level: ancestor.level,
      }));
      setBreadcrumbs(nextBreadcrumbs.length > 0 ? nextBreadcrumbs : [rootItem]);
      const nodeId = graphNodeId(result.kind, result.target_id);
      setSelection({ kind: result.kind, id: result.target_id, graphNodeId: nodeId });
      dispatch({ type: "select-node", nodeId });
      dispatch({ type: "highlight-element", elementId: nodeId });
      pendingFocusRef.current = nodeId;
      navigationStartedAtRef.current = clockNow();
      navigationLockRef.current = true;
      setSearchValue("");
    },
    [rootItem],
  );

  useEffect(() => {
    const graph = builtGraph.graph;
    if (!graph) return;
    const graphChanged = renderedGraphRef.current !== graph;
    if (graphChanged) renderedGraphRef.current = graph;
    const focusTarget = pendingFocusRef.current;
    if (!graphChanged && (!focusTarget || !graph.hasNode(focusTarget))) return;
    const settleCamera = async () => {
      if (focusTarget && graph.hasNode(focusTarget)) {
        await canvasRef.current?.focusNode(focusTarget, {
          ratio: 0.3,
          duration: 280,
        });
        pendingFocusRef.current = null;
      } else if (graphChanged) {
        await canvasRef.current?.reset();
      }
      navigationLockRef.current = false;
    };
    void settleCamera();
  }, [builtGraph.graph, selection?.graphNodeId]);

  const handleCanvasMetric = useCallback((metric: BrainCanvasMetric) => {
    setMetrics((current) => {
      if (metric.name === "first-paint" || metric.name === "graph-paint") {
        const navigationStartedAt = navigationStartedAtRef.current;
        const navigationMs =
          navigationStartedAt === null ? current.navigationMs : clockNow() - navigationStartedAt;
        if (navigationStartedAt !== null) navigationStartedAtRef.current = null;
        return {
          ...current,
          firstPaintMs: metric.durationMs,
          navigationMs,
        };
      }
      if (metric.name === "hover") {
        return { ...current, hoverMs: metric.durationMs };
      }
      return { ...current, navigationMs: metric.durationMs };
    });
  }, []);

  const handleCameraRatio = useCallback(
    (ratio: number) => {
      const previousRatio = previousCameraRatioRef.current;
      previousCameraRatioRef.current = ratio;
      let tier = semanticZoomTierForRatio(ratio);
      if (!graphContainsClusters) tier = "knowledge";
      else if (currentClusterId === null && tier === "themes") tier = "domains";
      dispatch({ type: "set-zoom-tier", tier });

      if (
        previousRatio > AUTO_ENTER_RATIO &&
        ratio <= AUTO_ENTER_RATIO &&
        shouldOfferDeeperNavigation(tier, graphContainsClusters)
      ) {
        const hovered = interaction.hoveredNodeId;
        const graph = builtGraph.graph;
        if (hovered && graph?.hasNode(hovered)) {
          const attributes = graph.getNodeAttributes(hovered);
          if (attributes.kind === "cluster" && attributes.clusterId) {
            void enterCluster({
              id: attributes.clusterId,
              label: attributes.label,
              level: graph.getAttribute("level"),
            });
          }
        }
      } else if (
        previousRatio < AUTO_EXIT_RATIO &&
        ratio >= AUTO_EXIT_RATIO &&
        breadcrumbs.length > 1 &&
        !navigationLockRef.current
      ) {
        navigateBreadcrumb(breadcrumbs.length - 2);
      }
    },
    [
      breadcrumbs.length,
      builtGraph.graph,
      currentClusterId,
      enterCluster,
      graphContainsClusters,
      interaction.hoveredNodeId,
      navigateBreadcrumb,
    ],
  );

  const openFullscreen = useCallback(async () => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    if (document.fullscreenElement === workspace) {
      await document.exitFullscreen();
    } else {
      await workspace.requestFullscreen();
    }
  }, []);

  if (statusQuery.isPending) {
    return (
      <section className="brain-page">
        <div className="brain-page-state" role="status">
          <span className="spinner" aria-hidden="true" />
          Chargement du cerveau…
        </div>
      </section>
    );
  }

  if (statusQuery.isError || !status || !availability) {
    return (
      <section className="page narrow-page">
        <p className="eyebrow">Cerveau indisponible</p>
        <h1>Impossible de charger la carte</h1>
        <p className="page-introduction">{getReadableError(statusQuery.error)}</p>
        <button className="button button-primary" type="button" onClick={() => void statusQuery.refetch()}>
          Réessayer
        </button>
      </section>
    );
  }

  if (!availability.canDisplayGraph) {
    const progress = status.active_job?.progress_percent;
    return (
      <section className="page narrow-page brain-empty-page">
        <p className="eyebrow">Carte sémantique</p>
        <h1>{availability.title}</h1>
        <p className="page-introduction">{availability.message}</p>
        {progress !== undefined ? (
          <div className="brain-build-progress" role="status">
            <span style={{ width: `${progress}%` }} />
            <strong>{progress}%</strong>
          </div>
        ) : null}
        <div className="brain-empty-actions">
          <Link className="button button-secondary" to="/parametres">
            Ouvrir les Paramètres
          </Link>
          {status.can_rebuild ? (
            <button
              className="button button-primary"
              type="button"
              disabled={rebuildMutation.isPending}
              onClick={() => rebuildMutation.mutate()}
            >
              {rebuildMutation.isPending ? "Lancement…" : "Construire le cerveau"}
            </button>
          ) : null}
        </div>
        {rebuildMutation.isError ? (
          <p className="alert alert-error" role="alert">
            {getReadableError(rebuildMutation.error)}
          </p>
        ) : null}
      </section>
    );
  }

  const panelLoading =
    selection?.kind === "cluster"
      ? clusterQuery.isPending
      : selection?.kind === "knowledge"
        ? knowledgeQuery.isPending
        : false;
  const panelError = clusterQuery.isError
    ? getReadableError(clusterQuery.error)
    : knowledgeQuery.isError
      ? getReadableError(knowledgeQuery.error)
      : null;
  const graphError = graphQuery.isError
    ? getReadableError(graphQuery.error)
    : builtGraph.error;
  const viewLabel =
    currentClusterId === null
      ? "Grands domaines"
      : graphContainsClusters
        ? "Sous-thèmes"
        : semanticZoomDescription("knowledge");

  return (
    <section className="brain-page">
      <header className="brain-page-header">
        <div>
          <p className="eyebrow">Carte sémantique</p>
          <h1>Cerveau</h1>
        </div>
        <p>
          Explorez vos connaissances par domaines, thèmes et relations. Aucun calcul IA n’est lancé pendant la navigation.
        </p>
      </header>

      {availability.message ? (
        <div className={`brain-status-banner is-${availability.tone}`} role="status">
          <div>
            <strong>{availability.title}</strong>
            <p>{availability.message}</p>
          </div>
          {status.active_job ? <span>{status.active_job.progress_percent}%</span> : null}
        </div>
      ) : null}

      <div
        ref={workspaceRef}
        className={`brain-workspace${selection ? " has-panel" : ""}`}
      >
        <div className="brain-map-shell">
          <div className="brain-map-topbar">
            <BrainBreadcrumb items={breadcrumbs.length > 0 ? breadcrumbs : [rootItem]} onNavigate={navigateBreadcrumb} />
            <BrainSearch
              value={searchValue}
              results={searchQuery.data?.items ?? []}
              loading={
                searchValue.trim() !== debouncedSearch || searchQuery.isFetching
              }
              error={searchQuery.isError ? getReadableError(searchQuery.error) : null}
              onChange={setSearchValue}
              onSelect={handleSearchSelection}
            />
          </div>

          {builtGraph.graph && presentation && !graphError ? (
            <BrainCanvas
              ref={canvasRef}
              graph={builtGraph.graph}
              className="brain-canvas"
              selectedNodeId={interaction.selectedNodeId}
              hoveredNodeId={interaction.hoveredNodeId}
              onNodeClick={selectGraphNode}
              onNodeDoubleClick={(nodeId) => {
                const graph = builtGraph.graph;
                if (!graph?.hasNode(nodeId)) return;
                const attributes = graph.getNodeAttributes(nodeId);
                if (attributes.kind === "cluster" && attributes.clusterId) {
                  void enterCluster({
                    id: attributes.clusterId,
                    label: attributes.label,
                    level: graph.getAttribute("level"),
                  });
                }
              }}
              onNodeHover={(nodeId) => dispatch({ type: "hover-node", nodeId })}
              onStageClick={clearSelection}
              onCameraRatioChange={handleCameraRatio}
              onMetric={handleCanvasMetric}
              nodeReducer={(nodeId, attributes) => {
                const state = presentation.nodes.get(nodeId);
                return state
                  ? {
                      ...attributes,
                      hidden: state.hidden,
                      color: state.color,
                      size: state.size,
                      forceLabel: state.forceLabel,
                      highlighted: state.highlighted,
                      zIndex: state.zIndex,
                    }
                  : attributes;
              }}
              edgeReducer={(edgeId, attributes) => {
                const state = presentation.edges.get(edgeId);
                return state
                  ? {
                      ...attributes,
                      hidden: state.hidden,
                      color: state.color,
                      size: state.size,
                      zIndex: state.zIndex,
                    }
                  : attributes;
              }}
              settings={{
                defaultNodeColor: "#9aa4bd",
                defaultEdgeColor: "rgba(160, 169, 194, 0.18)",
                labelColor: { color: "#e8ebf5" },
                labelFont: "Inter, system-ui, sans-serif",
                labelSize: 12,
                labelWeight: "600",
                labelDensity: 0.08,
                labelGridCellSize: 110,
                labelRenderedSizeThreshold: 7,
                hideEdgesOnMove: true,
                hideLabelsOnMove: true,
                renderEdgeLabels: false,
                stagePadding: 96,
                minCameraRatio: 0.04,
                maxCameraRatio: 4,
              }}
            />
          ) : null}

          {graphQuery.isPending ? (
            <div className="brain-canvas-state" role="status">
              <span className="spinner spinner-light" aria-hidden="true" />
              Chargement de la carte…
            </div>
          ) : null}
          {graphQuery.isFetching && !graphQuery.isPending ? (
            <div className="brain-level-loading" role="status">
              Ouverture du niveau…
            </div>
          ) : null}
          {graphError ? (
            <div className="brain-canvas-state is-error" role="alert">
              <strong>Impossible d’afficher ce niveau</strong>
              <p>{graphError}</p>
              <button type="button" onClick={() => void graphQuery.refetch()}>
                Réessayer
              </button>
            </div>
          ) : null}
          {builtGraph.graph?.order === 0 && !graphError ? (
            <div className="brain-canvas-state">
              <strong>Ce niveau ne contient aucune connaissance.</strong>
            </div>
          ) : null}

          <BrainToolbar
            viewLabel={viewLabel}
            metrics={metrics}
            onZoomIn={() => void canvasRef.current?.zoomIn()}
            onZoomOut={() => void canvasRef.current?.zoomOut()}
            onRecenter={() => void canvasRef.current?.recenter()}
            onGlobalView={() => {
              if (currentClusterId === null) {
                clearSelection();
                void canvasRef.current?.reset();
              } else {
                navigateBreadcrumb(0);
              }
            }}
            onFullscreen={() => void openFullscreen()}
          />
          <BrainLegend />
          {graphQuery.data?.data.truncated ? (
            <p className="brain-truncated-warning">
              Ce niveau est borné pour préserver la fluidité de l’affichage.
            </p>
          ) : null}
        </div>

        {selection ? (
          <BrainDetailsPanel
            cluster={selection.kind === "cluster" ? (clusterQuery.data ?? null) : null}
            knowledge={selection.kind === "knowledge" ? (knowledgeQuery.data ?? null) : null}
            loading={panelLoading}
            error={panelError}
            onClose={clearSelection}
            onEnterCluster={(cluster) =>
              void enterCluster({
                id: cluster.id,
                label: cluster.label,
                level: cluster.level,
              })
            }
            onSelectChild={(clusterId) => {
              const parent = clusterQuery.data;
              const child = parent?.children.find((item) => item.id === clusterId);
              if (!parent || !child) return;
              pendingFocusRef.current = graphNodeId("cluster", child.id);
              void enterCluster({
                id: parent.id,
                label: parent.label,
                level: parent.level,
              });
            }}
          />
        ) : null}
      </div>
    </section>
  );
}

import type Graph from "graphology";
import type { Attributes } from "graphology-types";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from "react";
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  ReactElement,
  RefAttributes,
} from "react";
import Sigma from "sigma";
import type { Settings } from "sigma/settings";
import type {
  CameraState,
  EdgeDisplayData,
  NodeDisplayData,
} from "sigma/types";

const CAMERA_ANIMATION_MS = 220;
const CAMERA_ZOOM_FACTOR = 1.5;

export type BrainCanvasMetricName =
  | "first-paint"
  | "graph-paint"
  | "hover"
  | "navigation";

export interface BrainCanvasMetric {
  name: BrainCanvasMetricName;
  durationMs: number;
  elementId: string | null;
}

export interface BrainCanvasRenderState<
  N extends Attributes,
  E extends Attributes,
  G extends Attributes,
> {
  graph: Graph<N, E, G>;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
}

export type BrainCanvasNodeReducer<
  N extends Attributes,
  E extends Attributes,
  G extends Attributes,
> = (
  nodeId: string,
  attributes: N,
  state: BrainCanvasRenderState<N, E, G>,
) => Partial<NodeDisplayData>;

export type BrainCanvasEdgeReducer<
  N extends Attributes,
  E extends Attributes,
  G extends Attributes,
> = (
  edgeId: string,
  attributes: E,
  state: BrainCanvasRenderState<N, E, G>,
) => Partial<EdgeDisplayData>;

export interface BrainCanvasFocusOptions {
  /** A lower Sigma camera ratio means a closer view. */
  ratio?: number;
  duration?: number;
}

export interface BrainCanvasHandle {
  zoomIn(): Promise<void>;
  zoomOut(): Promise<void>;
  /** Preserve the current zoom while moving the camera back to the graph centre. */
  recenter(): Promise<void>;
  /** Restore Sigma's complete global view (centre, zoom and angle). */
  reset(): Promise<void>;
  focusNode(nodeId: string, options?: BrainCanvasFocusOptions): Promise<boolean>;
  refresh(): void;
}

export interface BrainCanvasProps<
  N extends Attributes,
  E extends Attributes,
  G extends Attributes,
> {
  graph: Graph<N, E, G>;
  selectedNodeId?: string | null;
  hoveredNodeId?: string | null;
  onNodeClick?: (nodeId: string) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
  onStageClick?: () => void;
  onCameraChange?: (state: CameraState) => void;
  onCameraRatioChange?: (ratio: number) => void;
  onMetric?: (metric: BrainCanvasMetric) => void;
  nodeReducer?: BrainCanvasNodeReducer<N, E, G>;
  edgeReducer?: BrainCanvasEdgeReducer<N, E, G>;
  settings?: Omit<Partial<Settings<N, E, G>>, "nodeReducer" | "edgeReducer">;
  className?: string;
  style?: CSSProperties;
  ariaLabel?: string;
}

interface BrainCanvasCallbacks {
  onNodeClick: ((nodeId: string) => void) | undefined;
  onNodeDoubleClick: ((nodeId: string) => void) | undefined;
  onNodeHover: ((nodeId: string | null) => void) | undefined;
  onStageClick: (() => void) | undefined;
  onCameraChange: ((state: CameraState) => void) | undefined;
  onCameraRatioChange: ((ratio: number) => void) | undefined;
  onMetric: ((metric: BrainCanvasMetric) => void) | undefined;
}

function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

function numericAttribute(attributes: Attributes, name: string, fallback: number): number {
  const value = attributes[name];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function BrainCanvasInner<
  N extends Attributes,
  E extends Attributes,
  G extends Attributes,
>(
  {
    graph,
    selectedNodeId = null,
    hoveredNodeId = null,
    onNodeClick,
    onNodeDoubleClick,
    onNodeHover,
    onStageClick,
    onCameraChange,
    onCameraRatioChange,
    onMetric,
    nodeReducer,
    edgeReducer,
    settings,
    className,
    style,
    ariaLabel = "Carte interactive du second cerveau. Utilisez la souris ou le toucher pour vous déplacer et zoomer.",
  }: BrainCanvasProps<N, E, G>,
  forwardedRef: React.ForwardedRef<BrainCanvasHandle>,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<Sigma<N, E, G> | null>(null);
  const activeGraphRef = useRef(graph);
  const selectedNodeRef = useRef(selectedNodeId);
  const hoveredNodeRef = useRef(hoveredNodeId);
  const nodeReducerRef = useRef(nodeReducer);
  const edgeReducerRef = useRef(edgeReducer);
  const callbacksRef = useRef<BrainCanvasCallbacks>({
    onNodeClick: undefined,
    onNodeDoubleClick: undefined,
    onNodeHover: undefined,
    onStageClick: undefined,
    onCameraChange: undefined,
    onCameraRatioChange: undefined,
    onMetric: undefined,
  });
  const settingsRef = useRef(settings);
  const previousCameraRatioRef = useRef<number | null>(null);
  const pendingHoverMetricRef = useRef<{
    startedAt: number;
    elementId: string;
    armed: boolean;
  } | null>(null);

  selectedNodeRef.current = selectedNodeId;
  hoveredNodeRef.current = hoveredNodeId;
  nodeReducerRef.current = nodeReducer;
  edgeReducerRef.current = edgeReducer;
  settingsRef.current = settings;
  callbacksRef.current = {
    onNodeClick,
    onNodeDoubleClick,
    onNodeHover,
    onStageClick,
    onCameraChange,
    onCameraRatioChange,
    onMetric,
  };

  const reportMetric = useCallback(
    (name: BrainCanvasMetricName, startedAt: number, elementId: string | null) => {
      callbacksRef.current.onMetric?.({
        name,
        durationMs: Math.max(0, now() - startedAt),
        elementId,
      });
    },
    [],
  );

  const reduceNode = useCallback(
    (nodeId: string, attributes: N): Partial<NodeDisplayData> => {
      const interactionState: BrainCanvasRenderState<N, E, G> = {
        graph: activeGraphRef.current,
        selectedNodeId: selectedNodeRef.current,
        hoveredNodeId: hoveredNodeRef.current,
      };
      // Sigma reducers replace the complete attribute object at runtime. Spreading
      // the source data is therefore essential even though its public type says
      // that a reducer may return a partial display object.
      const customReducer = nodeReducerRef.current;
      if (customReducer !== undefined) {
        return { ...attributes, ...customReducer(nodeId, attributes, interactionState) };
      }
      const reduced = { ...attributes };
      // A persistent selection owns the neighbourhood emphasis. Hovering another
      // node still highlights that node, but must not replace the selected node's
      // direct relations.
      const focusId = interactionState.selectedNodeId ?? interactionState.hoveredNodeId;
      const graphForRender = interactionState.graph;
      const focusExists = focusId !== null && graphForRender.hasNode(focusId);
      const isSelected = nodeId === interactionState.selectedNodeId;
      const isHovered = nodeId === interactionState.hoveredNodeId;
      const isNeighbour =
        focusExists && focusId !== nodeId
          ? graphForRender.areNeighbors(focusId, nodeId)
          : false;
      const size = numericAttribute(reduced, "size", 1);

      if (isSelected || isHovered) {
        return {
          ...reduced,
          size: size * (isSelected ? 1.32 : 1.18),
          forceLabel: true,
          highlighted: true,
          zIndex: isSelected ? 4 : 3,
        };
      }
      if (focusExists && focusId !== nodeId && !isNeighbour) {
        return {
          ...reduced,
          color: "rgba(105, 111, 132, 0.2)",
          forceLabel: false,
          highlighted: false,
          zIndex: 0,
        };
      }
      if (isNeighbour) {
        return {
          ...reduced,
          highlighted: true,
          zIndex: 2,
        };
      }
      return reduced;
    },
    [],
  );

  const reduceEdge = useCallback(
    (edgeId: string, attributes: E): Partial<EdgeDisplayData> => {
      const interactionState: BrainCanvasRenderState<N, E, G> = {
        graph: activeGraphRef.current,
        selectedNodeId: selectedNodeRef.current,
        hoveredNodeId: hoveredNodeRef.current,
      };
      const customReducer = edgeReducerRef.current;
      if (customReducer !== undefined) {
        return { ...attributes, ...customReducer(edgeId, attributes, interactionState) };
      }
      const reduced = { ...attributes };
      const focusId = interactionState.selectedNodeId ?? interactionState.hoveredNodeId;
      if (focusId === null || !interactionState.graph.hasNode(focusId)) {
        return reduced;
      }

      const [source, target] = interactionState.graph.extremities(edgeId);
      const isIncident = source === focusId || target === focusId;
      const size = numericAttribute(reduced, "size", 1);
      return isIncident
        ? {
            ...reduced,
            size: Math.max(1.25, size * 1.75),
            highlighted: true,
            zIndex: 3,
          }
        : {
            ...reduced,
            size: Math.max(0.25, size * 0.45),
            color: "rgba(105, 111, 132, 0.07)",
            highlighted: false,
            zIndex: 0,
          };
    },
    [],
  );

  const measureCameraAnimation = useCallback(
    async (operation: () => Promise<void>, elementId: string | null = null) => {
      const startedAt = now();
      await operation();
      reportMetric("navigation", startedAt, elementId);
    },
    [reportMetric],
  );

  const zoomIn = useCallback(async () => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    await measureCameraAnimation(() =>
      renderer.getCamera().animatedZoom({
        factor: CAMERA_ZOOM_FACTOR,
        duration: CAMERA_ANIMATION_MS,
      }),
    );
  }, [measureCameraAnimation]);

  const zoomOut = useCallback(async () => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    await measureCameraAnimation(() =>
      renderer.getCamera().animatedUnzoom({
        factor: CAMERA_ZOOM_FACTOR,
        duration: CAMERA_ANIMATION_MS,
      }),
    );
  }, [measureCameraAnimation]);

  const recenter = useCallback(async () => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    const currentRatio = renderer.getCamera().getState().ratio;
    await measureCameraAnimation(() =>
      renderer.getCamera().animate(
        { x: 0.5, y: 0.5, angle: 0, ratio: currentRatio },
        { duration: CAMERA_ANIMATION_MS },
      ),
    );
  }, [measureCameraAnimation]);

  const reset = useCallback(async () => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    await measureCameraAnimation(() =>
      renderer.getCamera().animatedReset({ duration: CAMERA_ANIMATION_MS }),
    );
  }, [measureCameraAnimation]);

  const focusNode = useCallback(
    async (
      nodeId: string,
      options: BrainCanvasFocusOptions = {},
    ): Promise<boolean> => {
      const renderer = rendererRef.current;
      if (renderer === null || !renderer.getGraph().hasNode(nodeId)) return false;
      const displayData = renderer.getNodeDisplayData(nodeId);
      if (displayData === undefined) return false;
      const currentRatio = renderer.getCamera().getState().ratio;
      const ratio = Math.max(
        0.02,
        Math.min(1, options.ratio ?? Math.min(currentRatio, 0.32)),
      );
      await measureCameraAnimation(
        () =>
          renderer.getCamera().animate(
            { x: displayData.x, y: displayData.y, ratio },
            { duration: options.duration ?? CAMERA_ANIMATION_MS },
          ),
        nodeId,
      );
      return true;
    },
    [measureCameraAnimation],
  );

  const refresh = useCallback(() => {
    rendererRef.current?.scheduleRefresh();
  }, []);

  const armPendingHoverMetric = useCallback((renderer: Sigma<N, E, G>) => {
    const pendingMetric = pendingHoverMetricRef.current;
    if (pendingMetric !== null && !pendingMetric.armed) {
      pendingMetric.armed = true;
      renderer.once("afterRender", () => {
        if (pendingHoverMetricRef.current === pendingMetric) {
          pendingHoverMetricRef.current = null;
        }
        reportMetric("hover", pendingMetric.startedAt, pendingMetric.elementId);
      });
    }
  }, [reportMetric]);

  const refreshInteraction = useCallback(() => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    armPendingHoverMetric(renderer);
    renderer.scheduleRefresh();
  }, [armPendingHoverMetric]);

  useImperativeHandle(
    forwardedRef,
    () => ({ zoomIn, zoomOut, recenter, reset, focusNode, refresh }),
    [focusNode, recenter, refresh, reset, zoomIn, zoomOut],
  );

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (container === null) return;

    const createdAt = now();
    const renderer = new Sigma<N, E, G>(activeGraphRef.current, container, {
      ...settingsRef.current,
      nodeReducer: reduceNode,
      edgeReducer: reduceEdge,
      enableCameraPanning: true,
      enableCameraZooming: true,
      enableCameraRotation: false,
      zIndex: true,
    });
    rendererRef.current = renderer;

    const handleClickNode = ({ node }: { node: string }) => {
      callbacksRef.current.onNodeClick?.(node);
    };
    const handleDoubleClickNode = (payload: {
      node: string;
      preventSigmaDefault(): void;
    }) => {
      if (callbacksRef.current.onNodeDoubleClick !== undefined) {
        payload.preventSigmaDefault();
        callbacksRef.current.onNodeDoubleClick(payload.node);
      }
    };
    const handleEnterNode = ({ node }: { node: string }) => {
      pendingHoverMetricRef.current = {
        startedAt: now(),
        elementId: node,
        armed: false,
      };
      armPendingHoverMetric(renderer);
      hoveredNodeRef.current = node;
      callbacksRef.current.onNodeHover?.(node);
      if (
        nodeReducerRef.current === undefined &&
        edgeReducerRef.current === undefined
      ) {
        refreshInteraction();
      }
    };
    const handleLeaveNode = ({ node }: { node: string }) => {
      pendingHoverMetricRef.current = {
        startedAt: now(),
        elementId: node,
        armed: false,
      };
      armPendingHoverMetric(renderer);
      if (hoveredNodeRef.current === node) hoveredNodeRef.current = null;
      callbacksRef.current.onNodeHover?.(null);
      if (
        nodeReducerRef.current === undefined &&
        edgeReducerRef.current === undefined
      ) {
        refreshInteraction();
      }
    };
    const handleClickStage = () => {
      callbacksRef.current.onStageClick?.();
    };
    const handleCameraUpdate = (state: CameraState) => {
      callbacksRef.current.onCameraChange?.(state);
      const previousRatio = previousCameraRatioRef.current;
      if (previousRatio === null || Math.abs(previousRatio - state.ratio) > 0.0001) {
        previousCameraRatioRef.current = state.ratio;
        callbacksRef.current.onCameraRatioChange?.(state.ratio);
      }
    };

    renderer.on("clickNode", handleClickNode);
    renderer.on("doubleClickNode", handleDoubleClickNode);
    renderer.on("enterNode", handleEnterNode);
    renderer.on("leaveNode", handleLeaveNode);
    renderer.on("clickStage", handleClickStage);
    renderer.getCamera().on("updated", handleCameraUpdate);

    renderer.once("afterRender", () => reportMetric("first-paint", createdAt, null));
    renderer.scheduleRefresh();

    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => {
            renderer.resize().scheduleRender();
          });
    resizeObserver?.observe(container);

    return () => {
      resizeObserver?.disconnect();
      renderer.off("clickNode", handleClickNode);
      renderer.off("doubleClickNode", handleDoubleClickNode);
      renderer.off("enterNode", handleEnterNode);
      renderer.off("leaveNode", handleLeaveNode);
      renderer.off("clickStage", handleClickStage);
      renderer.getCamera().off("updated", handleCameraUpdate);
      renderer.kill();
      if (rendererRef.current === renderer) rendererRef.current = null;
    };
  }, [
    armPendingHoverMetric,
    reduceEdge,
    reduceNode,
    refreshInteraction,
    reportMetric,
  ]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (renderer === null || renderer.getGraph() === graph) return;
    const startedAt = now();
    activeGraphRef.current = graph;
    renderer.once("afterRender", () => reportMetric("graph-paint", startedAt, null));
    renderer.setGraph(graph);
  }, [graph, reportMetric]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (renderer === null) return;
    renderer.setSettings({
      ...settings,
      nodeReducer: reduceNode,
      edgeReducer: reduceEdge,
      enableCameraPanning: true,
      enableCameraZooming: true,
      enableCameraRotation: false,
      zIndex: true,
    });
    renderer.scheduleRefresh();
  }, [edgeReducer, nodeReducer, reduceEdge, reduceNode, settings]);

  useEffect(() => {
    refreshInteraction();
  }, [hoveredNodeId, refreshInteraction, selectedNodeId]);

  function handleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      void zoomIn();
    } else if (event.key === "-") {
      event.preventDefault();
      void zoomOut();
    } else if (event.key === "0" || event.key === "Home") {
      event.preventDefault();
      void reset();
    }
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        touchAction: "none",
        ...style,
      }}
      role="application"
      aria-label={ariaLabel}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    />
  );
}

const ForwardedBrainCanvas = forwardRef(BrainCanvasInner);
ForwardedBrainCanvas.displayName = "BrainCanvas";

export const BrainCanvas = ForwardedBrainCanvas as <
  N extends Attributes = Attributes,
  E extends Attributes = Attributes,
  G extends Attributes = Attributes,
>(
  props: BrainCanvasProps<N, E, G> & RefAttributes<BrainCanvasHandle>,
) => ReactElement;

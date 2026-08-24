// @vitest-environment jsdom

import { UndirectedGraph } from "graphology";
import { act, fireEvent, render } from "@testing-library/react";
import { createRef } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  BrainCanvas,
  type BrainCanvasHandle,
  type BrainCanvasMetric,
} from "../BrainCanvas";

const sigmaMock = vi.hoisted(() => ({ instances: [] as unknown[] }));

vi.mock("sigma", () => {
  type Listener = (...arguments_: unknown[]) => void;

  class FakeEmitter {
    listeners = new Map<string, Set<Listener>>();

    on(event: string, listener: Listener) {
      const eventListeners = this.listeners.get(event) ?? new Set<Listener>();
      eventListeners.add(listener);
      this.listeners.set(event, eventListeners);
      return this;
    }

    once(event: string, listener: Listener) {
      const wrapper: Listener = (...arguments_) => {
        this.off(event, wrapper);
        listener(...arguments_);
      };
      return this.on(event, wrapper);
    }

    off(event: string, listener: Listener) {
      this.listeners.get(event)?.delete(listener);
      return this;
    }

    emit(event: string, ...arguments_: unknown[]) {
      for (const listener of [...(this.listeners.get(event) ?? [])]) {
        listener(...arguments_);
      }
    }
  }

  class FakeCamera extends FakeEmitter {
    state = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };
    operations: string[] = [];

    getState() {
      return { ...this.state };
    }

    animate(nextState: Partial<typeof this.state>) {
      this.operations.push("animate");
      this.state = { ...this.state, ...nextState };
      this.emit("updated", this.getState());
      return Promise.resolve();
    }

    animatedZoom() {
      this.operations.push("zoom");
      this.state.ratio /= 1.5;
      this.emit("updated", this.getState());
      return Promise.resolve();
    }

    animatedUnzoom() {
      this.operations.push("unzoom");
      this.state.ratio *= 1.5;
      this.emit("updated", this.getState());
      return Promise.resolve();
    }

    animatedReset() {
      this.operations.push("reset");
      this.state = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };
      this.emit("updated", this.getState());
      return Promise.resolve();
    }
  }

  class FakeSigma extends FakeEmitter {
    camera = new FakeCamera();
    graph: UndirectedGraph;
    settings: Record<string, unknown>;
    setGraphCalls: unknown[] = [];
    refreshCount = 0;
    killed = false;

    constructor(graph: UndirectedGraph, _container: HTMLElement, settings: object) {
      super();
      this.graph = graph;
      this.settings = { ...settings };
      sigmaMock.instances.push(this);
    }

    getGraph() {
      return this.graph;
    }

    setGraph(graph: UndirectedGraph) {
      this.graph = graph;
      this.setGraphCalls.push(graph);
      this.emit("afterRender");
      return this;
    }

    getCamera() {
      return this.camera;
    }

    getNodeDisplayData(nodeId: string) {
      if (!this.graph.hasNode(nodeId)) return undefined;
      return this.graph.getNodeAttributes(nodeId);
    }

    setSettings(settings: object) {
      this.settings = { ...this.settings, ...settings };
      return this;
    }

    scheduleRefresh() {
      this.refreshCount += 1;
      this.emit("afterRender");
      return this;
    }

    scheduleRender() {
      this.emit("afterRender");
      return this;
    }

    resize() {
      return this;
    }

    kill() {
      this.killed = true;
      this.emit("kill");
      this.listeners.clear();
    }
  }

  return { default: FakeSigma };
});

interface NodeAttributes {
  x: number;
  y: number;
  size: number;
  label: string;
  color: string;
}

interface EdgeAttributes {
  size: number;
  color: string;
}

type TestGraph = UndirectedGraph<NodeAttributes, EdgeAttributes>;

interface FakeSigmaView {
  camera: {
    operations: string[];
    state: { x: number; y: number; ratio: number; angle: number };
    emit(event: string, payload: unknown): void;
  };
  graph: TestGraph;
  settings: {
    nodeReducer?: (nodeId: string, attributes: NodeAttributes) => Record<string, unknown>;
    edgeReducer?: (edgeId: string, attributes: EdgeAttributes) => Record<string, unknown>;
  };
  setGraphCalls: unknown[];
  refreshCount: number;
  killed: boolean;
  emit(event: string, payload?: unknown): void;
}

function createGraph(suffix = ""): TestGraph {
  const graph = new UndirectedGraph<NodeAttributes, EdgeAttributes>();
  graph.addNode(`a${suffix}`, {
    x: 0,
    y: 0,
    size: 4,
    label: "Alpha",
    color: "#8f7cff",
  });
  graph.addNode(`b${suffix}`, {
    x: 1,
    y: 0,
    size: 3,
    label: "Beta",
    color: "#8f7cff",
  });
  graph.addNode(`c${suffix}`, {
    x: 0,
    y: 1,
    size: 3,
    label: "Gamma",
    color: "#62b3ee",
  });
  graph.addUndirectedEdgeWithKey(`ab${suffix}`, `a${suffix}`, `b${suffix}`, {
    size: 1,
    color: "#6f7490",
  });
  return graph;
}

function currentRenderer(): FakeSigmaView {
  return sigmaMock.instances.at(-1) as FakeSigmaView;
}

describe("BrainCanvas", () => {
  beforeEach(() => {
    sigmaMock.instances.length = 0;
  });

  it("conserve une instance Sigma, remplace seulement le graphe et nettoie WebGL", () => {
    const firstGraph = createGraph();
    const secondGraph = createGraph("-next");
    const metrics: BrainCanvasMetric[] = [];
    const view = render(
      <BrainCanvas graph={firstGraph} onMetric={(metric) => metrics.push(metric)} />,
    );
    const renderer = currentRenderer();

    expect(sigmaMock.instances).toHaveLength(1);
    expect(metrics.some((metric) => metric.name === "first-paint")).toBe(true);

    view.rerender(
      <BrainCanvas graph={secondGraph} onMetric={(metric) => metrics.push(metric)} />,
    );

    expect(sigmaMock.instances).toHaveLength(1);
    expect(renderer.setGraphCalls).toEqual([secondGraph]);
    expect(renderer.graph).toBe(secondGraph);
    expect(metrics.some((metric) => metric.name === "graph-paint")).toBe(true);

    view.unmount();
    expect(renderer.killed).toBe(true);
    expect(renderer.camera.operations).toEqual([]);
  });

  it("relaie les interactions et expose des contrôles de caméra animés", async () => {
    const graph = createGraph();
    const canvasRef = createRef<BrainCanvasHandle>();
    const onClick = vi.fn();
    const onDoubleClick = vi.fn();
    const onHover = vi.fn();
    const onStageClick = vi.fn();
    const onRatio = vi.fn();
    const metrics: BrainCanvasMetric[] = [];
    const { getByRole } = render(
      <BrainCanvas
        ref={canvasRef}
        graph={graph}
        onNodeClick={onClick}
        onNodeDoubleClick={onDoubleClick}
        onNodeHover={onHover}
        onStageClick={onStageClick}
        onCameraRatioChange={onRatio}
        onMetric={(metric) => metrics.push(metric)}
      />,
    );
    const renderer = currentRenderer();
    const preventSigmaDefault = vi.fn();

    act(() => {
      renderer.emit("clickNode", { node: "a" });
      renderer.emit("doubleClickNode", { node: "a", preventSigmaDefault });
      renderer.emit("enterNode", { node: "a" });
      renderer.emit("leaveNode", { node: "a" });
      renderer.emit("clickStage", {});
    });

    expect(onClick).toHaveBeenCalledWith("a");
    expect(onDoubleClick).toHaveBeenCalledWith("a");
    expect(preventSigmaDefault).toHaveBeenCalledOnce();
    expect(onHover.mock.calls).toEqual([["a"], [null]]);
    expect(onStageClick).toHaveBeenCalledOnce();
    expect(metrics.filter((metric) => metric.name === "hover")).toHaveLength(2);

    await act(async () => {
      await canvasRef.current?.zoomIn();
      await canvasRef.current?.zoomOut();
      await canvasRef.current?.recenter();
      await canvasRef.current?.reset();
      expect(await canvasRef.current?.focusNode("a", { ratio: 0.2 })).toBe(true);
      expect(await canvasRef.current?.focusNode("absent")).toBe(false);
    });

    expect(renderer.camera.operations).toEqual([
      "zoom",
      "unzoom",
      "animate",
      "reset",
      "animate",
    ]);
    expect(renderer.camera.state).toMatchObject({ x: 0, y: 0, ratio: 0.2 });
    expect(onRatio).toHaveBeenCalled();
    expect(metrics.filter((metric) => metric.name === "navigation")).toHaveLength(5);

    fireEvent.keyDown(getByRole("application"), { key: "+" });
    await act(async () => Promise.resolve());
    expect(renderer.camera.operations.at(-1)).toBe("zoom");
  });

  it("met en avant la sélection et ses voisins sans muter Graphology", () => {
    const graph = createGraph();
    const originalAttributes = graph.getNodeAttributes("c");
    const view = render(
      <BrainCanvas graph={graph} selectedNodeId="a" hoveredNodeId={null} />,
    );
    const renderer = currentRenderer();
    const nodeReducer = renderer.settings.nodeReducer;
    const edgeReducer = renderer.settings.edgeReducer;

    expect(nodeReducer).toBeDefined();
    expect(edgeReducer).toBeDefined();
    expect(nodeReducer?.("a", graph.getNodeAttributes("a"))).toMatchObject({
      highlighted: true,
      forceLabel: true,
      size: 5.28,
      x: 0,
      y: 0,
    });
    expect(nodeReducer?.("b", graph.getNodeAttributes("b"))).toMatchObject({
      highlighted: true,
      color: "#8f7cff",
    });
    expect(nodeReducer?.("c", graph.getNodeAttributes("c"))).toMatchObject({
      highlighted: false,
      forceLabel: false,
      color: "rgba(105, 111, 132, 0.2)",
    });
    expect(edgeReducer?.("ab", graph.getEdgeAttributes("ab"))).toMatchObject({
      highlighted: true,
      size: 1.75,
    });

    view.rerender(
      <BrainCanvas graph={graph} selectedNodeId="a" hoveredNodeId="c" />,
    );
    expect(nodeReducer?.("c", graph.getNodeAttributes("c"))).toMatchObject({
      highlighted: true,
      forceLabel: true,
    });
    expect(nodeReducer?.("b", graph.getNodeAttributes("b"))).toMatchObject({
      highlighted: true,
      color: "#8f7cff",
    });
    expect(graph.getNodeAttributes("c")).toEqual(originalAttributes);

    view.rerender(
      <BrainCanvas
        graph={graph}
        selectedNodeId="a"
        hoveredNodeId={null}
        nodeReducer={() => ({ size: 9, highlighted: false, forceLabel: false })}
      />,
    );
    expect(nodeReducer?.("a", graph.getNodeAttributes("a"))).toMatchObject({
      size: 9,
      highlighted: false,
      forceLabel: false,
      x: 0,
      y: 0,
    });
  });
});

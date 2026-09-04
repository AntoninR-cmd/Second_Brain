// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { forwardRef, useImperativeHandle } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  BrainCluster,
  BrainClusterDetail,
  BrainGraphResponse,
  BrainProfile,
  BrainSearchResponse,
  BrainStatus,
} from "../../../api/types";
import { BrainPage } from "../../../pages/BrainPage";
import type { BrainCanvasHandle } from "../BrainCanvas";

const api = vi.hoisted(() => ({
  getBrainStatus: vi.fn(),
  getBrainClusters: vi.fn(),
  getBrainGraph: vi.fn(),
  getBrainCluster: vi.fn(),
  getKnowledgeNode: vi.fn(),
  rebuildBrain: vi.fn(),
  searchBrain: vi.fn(),
}));

const camera = vi.hoisted(() => ({
  focusNode: vi.fn(async () => true),
  zoomIn: vi.fn(async () => undefined),
  zoomOut: vi.fn(async () => undefined),
  recenter: vi.fn(async () => undefined),
  reset: vi.fn(async () => undefined),
  refresh: vi.fn(),
}));

vi.mock("../../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../api/client")>();
  return { ...actual, ...api };
});

vi.mock("../BrainCanvas", () => ({
  BrainCanvas: forwardRef<BrainCanvasHandle>(function MockBrainCanvas(_props, ref) {
    useImperativeHandle(ref, () => camera);
    return <div data-testid="brain-canvas" />;
  }),
}));

const profile: BrainProfile = {
  id: "profile",
  logical_generation: 1,
  status: "ready",
  embedding_profile_id: "embedding-profile",
  embedding_provider: "ollama",
  embedding_model_name: "qwen3-embedding:0.6b",
  embedding_model_digest: null,
  embedding_dimensions: 1024,
  embedding_semantic_text_version: "v1",
  embedding_logical_generation: 1,
  algorithm_version: "brain-v1",
  knowledge_node_count: 66,
  cluster_count: 9,
  edge_count: 207,
  unassigned_node_count: 0,
  cluster_counts_by_level: { "0": 1, "1": 2, "2": 6 },
  similarity: { minimum: 0.1, mean: 0.5, median: 0.5, maximum: 0.9 },
  cluster_sizes: { minimum: 5, mean: 11, maximum: 20 },
  relations_duration_ms: 1,
  clustering_duration_ms: 1,
  umap_duration_ms: 1,
  labeling_duration_ms: 1,
  total_duration_ms: 4,
  label_strategy: "deterministic",
  label_model_name: null,
  label_model_digest: null,
  created_at: "2026-08-24T10:00:00Z",
  completed_at: "2026-08-24T10:00:04Z",
  activated_at: "2026-08-24T10:00:04Z",
  error_message: null,
};

const readyStatus: BrainStatus = {
  state: "ready",
  active_profile: profile,
  building_profile: null,
  active_job: null,
  latest_job: null,
  stale_reasons: [],
  can_rebuild: true,
  can_relabel: true,
  error: null,
};

const rootCluster: BrainCluster = {
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
};

const domainCluster: BrainClusterDetail = {
  id: "domain",
  parent_id: "root",
  level: 1,
  label: "Principes d'entrainement en musculation",
  description: null,
  label_source: "deterministic",
  member_count: 48,
  representative_knowledge_node_ids: [],
  x: 0.2,
  y: -0.3,
  child_count: 1,
  children: [],
  knowledge_nodes: [],
};

const rootGraph: BrainGraphResponse = {
  profile_id: "profile",
  level: 1,
  parent_cluster_id: "root",
  nodes: [
    {
      id: "cluster:domain",
      kind: "cluster",
      label: domainCluster.label,
      x: domainCluster.x,
      y: domainCluster.y,
      size: domainCluster.member_count,
      cluster_id: domainCluster.id,
      knowledge_node_id: null,
      source_id: null,
      tags: [],
      href: null,
    },
  ],
  edges: [],
  truncated: false,
};

const domainGraph: BrainGraphResponse = {
  profile_id: "profile",
  level: 2,
  parent_cluster_id: "domain",
  nodes: [],
  edges: [],
  truncated: false,
};

const searchResponse: BrainSearchResponse = {
  profile_id: "profile",
  query: "musculation",
  items: [
    {
      kind: "cluster",
      target_id: domainCluster.id,
      label: domainCluster.label,
      level: domainCluster.level,
      cluster_id: domainCluster.id,
      x: domainCluster.x,
      y: domainCluster.y,
      member_count: domainCluster.member_count,
      tags: [],
      source_id: null,
      source_title: null,
      href: null,
      ancestors: [{ id: "root", label: "Second Brain", level: 0 }],
    },
  ],
};

function renderPage(): QueryClient {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <BrainPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return queryClient;
}

async function selectDomainFromSearch(): Promise<void> {
  const input = await screen.findByRole("searchbox", {
    name: /Rechercher un th.me ou une connaissance/,
  });
  fireEvent.change(input, { target: { value: "musculation" } });
  const option = await screen.findByRole("option");
  fireEvent.click(option.querySelector("button") ?? option);
  await waitFor(() => expect(camera.focusNode).toHaveBeenCalled());
}

describe("BrainPage navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getBrainStatus.mockResolvedValue(readyStatus);
    api.getBrainClusters.mockResolvedValue([rootCluster]);
    api.getBrainCluster.mockResolvedValue(domainCluster);
    api.getBrainGraph.mockImplementation(
      async (input?: { clusterId?: string }) =>
        input?.clusterId === domainCluster.id ? domainGraph : rootGraph,
    );
    api.searchBrain.mockResolvedValue(searchResponse);
  });

  afterEach(() => {
    cleanup();
  });

  it("deverrouille la camera apres deux selections identiques sur le meme graphe", async () => {
    const queryClient = renderPage();

    await selectDomainFromSearch();
    const firstFocusCount = camera.focusNode.mock.calls.length;
    await selectDomainFromSearch();
    await waitFor(() =>
      expect(camera.focusNode.mock.calls.length).toBeGreaterThan(firstFocusCount),
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /Entrer dans ce th.me/ }),
    );
    await waitFor(() =>
      expect(api.getBrainGraph).toHaveBeenCalledWith({ clusterId: "domain" }),
    );

    queryClient.clear();
  });
});

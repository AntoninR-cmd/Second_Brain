// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  BrainClusterDetail,
  BrainSearchResult,
  BrainStatus,
  KnowledgeNodeDetail,
} from "../../../api/types";
import { BrainBreadcrumb } from "../BrainBreadcrumb";
import { BrainDetailsPanel } from "../BrainDetailsPanel";
import { BrainSearch } from "../BrainSearch";
import { getBrainAvailability } from "../status";

afterEach(cleanup);

function status(state: BrainStatus["state"], withProfile = false): BrainStatus {
  return {
    state,
    active_profile: withProfile
      ? ({ id: "profile", status: state === "stale" ? "stale" : "ready" } as BrainStatus["active_profile"])
      : null,
    building_profile: null,
    active_job: null,
    latest_job: null,
    stale_reasons: [],
    can_rebuild: true,
    can_relabel: false,
    error: state === "error" ? "Layout corrompu" : null,
  };
}

const cluster: BrainClusterDetail = {
  id: "theme",
  parent_id: "domain",
  level: 2,
  label: "Organisation des séances",
  description: "Ordre et structure des exercices.",
  label_source: "ollama",
  member_count: 13,
  representative_knowledge_node_ids: ["knowledge"],
  x: 0.2,
  y: -0.3,
  child_count: 1,
  children: [
    {
      id: "subtheme",
      parent_id: "theme",
      level: 3,
      label: "Priorité des exercices",
      description: null,
      label_source: "deterministic",
      member_count: 4,
      representative_knowledge_node_ids: [],
      x: 0.1,
      y: -0.1,
      child_count: 0,
    },
  ],
  knowledge_nodes: [],
};

const knowledge: KnowledgeNodeDetail = {
  id: "knowledge",
  source_id: "source",
  title: "Éviter la redondance musculaire",
  content: "Deux exercices identiques peuvent produire une fatigue inutile.",
  tags: ["musculation", "fatigue"],
  evidence_count: 1,
  created_at: "2026-08-24T10:00:00Z",
  updated_at: "2026-08-24T10:00:00Z",
  source: {
    id: "source",
    type: "srt",
    title: "Programmer son entraînement",
    author: "Auteur test",
    original_filename: "training.srt",
    original_file_path: "originals/source/original.srt",
  },
  evidences: [
    {
      id: "evidence",
      passage_id: "passage",
      passage_index: 4,
      original_excerpt: "Évitez de répéter deux mouvements presque identiques.",
      start_ms: 1_000,
      end_ms: 3_250,
      first_segment_index: 8,
      last_segment_index: 10,
      char_start: null,
      char_end: null,
    },
  ],
};

const searchResult: BrainSearchResult = {
  kind: "knowledge",
  target_id: "knowledge",
  label: knowledge.title,
  level: null,
  cluster_id: "theme",
  x: 0.2,
  y: -0.3,
  member_count: null,
  tags: knowledge.tags,
  source_id: "source",
  source_title: knowledge.source.title,
  href: "/connaissances/knowledge",
  ancestors: [
    { id: "root", label: "Second Brain", level: 0 },
    { id: "domain", label: "Musculation", level: 1 },
    { id: "theme", label: cluster.label, level: 2 },
  ],
};

describe("états du BrainProfile", () => {
  it("affiche le profil ready", () => {
    expect(getBrainAvailability(status("ready", true))).toMatchObject({
      canDisplayGraph: true,
      tone: "ready",
    });
  });

  it("conserve le graphe stale avec un avertissement", () => {
    expect(getBrainAvailability(status("stale", true))).toMatchObject({
      canDisplayGraph: true,
      tone: "warning",
    });
  });

  it("conserve l'ancien profil pendant une reconstruction", () => {
    expect(getBrainAvailability(status("building", true))).toMatchObject({
      canDisplayGraph: true,
      title: "Reconstruction en cours",
    });
  });

  it("bloque seulement quand la première construction n'est pas terminée", () => {
    expect(getBrainAvailability(status("building"))).toMatchObject({
      canDisplayGraph: false,
      tone: "info",
    });
  });

  it("rend l'erreur explicite sans profil utilisable", () => {
    expect(getBrainAvailability(status("error"))).toMatchObject({
      canDisplayGraph: false,
      message: "Layout corrompu",
    });
  });
});

describe("navigation et recherche", () => {
  it("permet de revenir par le fil d'Ariane", () => {
    const onNavigate = vi.fn();
    render(
      <BrainBreadcrumb
        items={[
          { id: "root", label: "Second Brain", level: 0 },
          { id: "domain", label: "Musculation", level: 1 },
          { id: "theme", label: cluster.label, level: 2 },
        ]}
        onNavigate={onNavigate}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Musculation" }));
    expect(onNavigate).toHaveBeenCalledWith(1);
    expect(screen.getByRole("button", { name: cluster.label })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("sélectionne un résultat au clavier", () => {
    const onSelect = vi.fn();
    render(
      <BrainSearch
        value="redondance"
        results={[searchResult]}
        loading={false}
        error={null}
        onChange={vi.fn()}
        onSelect={onSelect}
      />,
    );
    fireEvent.keyDown(
      screen.getByRole("searchbox", {
        name: "Rechercher un thème ou une connaissance",
      }),
      { key: "Enter" },
    );
    expect(onSelect).toHaveBeenCalledWith(searchResult);
  });

  it("annonce une recherche sans résultat", () => {
    render(
      <BrainSearch
        value="inconnu"
        results={[]}
        loading={false}
        error={null}
        onChange={vi.fn()}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("Aucun résultat dans ce cerveau.")).toBeVisible();
  });
});

describe("panneau latéral", () => {
  it("affiche la connaissance, sa source et sa preuve SRT", () => {
    render(
      <MemoryRouter>
        <BrainDetailsPanel
          cluster={null}
          knowledge={knowledge}
          loading={false}
          error={null}
          onClose={vi.fn()}
          onEnterCluster={vi.fn()}
          onSelectChild={vi.fn()}
        />
      </MemoryRouter>,
    );
    expect(screen.getByText(knowledge.content)).toBeVisible();
    expect(screen.getByText("#musculation")).toBeVisible();
    expect(screen.getByText(knowledge.source.title)).toBeVisible();
    expect(screen.getByText("Auteur test")).toBeVisible();
    expect(screen.getByText("00:00:01,000 → 00:00:03,250")).toBeVisible();
    expect(screen.getByRole("link", { name: /Ouvrir la connaissance/ })).toHaveAttribute(
      "href",
      "/connaissances/knowledge",
    );
  });

  it("affiche les sous-thèmes et les actions d'un cluster", () => {
    const onEnter = vi.fn();
    const onSelectChild = vi.fn();
    render(
      <MemoryRouter>
        <BrainDetailsPanel
          cluster={cluster}
          knowledge={null}
          loading={false}
          error={null}
          onClose={vi.fn()}
          onEnterCluster={onEnter}
          onSelectChild={onSelectChild}
        />
      </MemoryRouter>,
    );
    expect(screen.getByText("13")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /Priorité des exercices/ }));
    expect(onSelectChild).toHaveBeenCalledWith("subtheme");
    fireEvent.click(screen.getByRole("button", { name: /Entrer dans ce thème/ }));
    expect(onEnter).toHaveBeenCalledWith(cluster);
  });
});

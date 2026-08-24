import { lazy, Suspense } from "react";
import { createBrowserRouter } from "react-router-dom";

import { AppShell } from "../components/layout/AppShell";
import { AddPage } from "../pages/AddPage";
import { DashboardPage } from "../pages/DashboardPage";
import { KnowledgeDetailPage } from "../pages/KnowledgeDetailPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { SearchPage } from "../pages/SearchPage";
import { SettingsPage } from "../pages/SettingsPage";
import { SourceDetailPage } from "../pages/SourceDetailPage";
import { SourcesPage } from "../pages/SourcesPage";

const BrainPage = lazy(() =>
  import("../pages/BrainPage").then((module) => ({ default: module.BrainPage })),
);

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "ajouter", element: <AddPage /> },
      {
        path: "cerveau",
        element: (
          <Suspense
            fallback={
              <div className="loading-state page-loading" role="status">
                <span className="spinner" aria-hidden="true" />
                Chargement de la carte…
              </div>
            }
          >
            <BrainPage />
          </Suspense>
        ),
      },
      { path: "sources", element: <SourcesPage /> },
      { path: "sources/:sourceId", element: <SourceDetailPage /> },
      { path: "connaissances/:nodeId", element: <KnowledgeDetailPage /> },
      { path: "recherche", element: <SearchPage /> },
      { path: "parametres", element: <SettingsPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

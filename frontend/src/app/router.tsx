import { createBrowserRouter } from "react-router-dom";

import { AppShell } from "../components/layout/AppShell";
import { AddPage } from "../pages/AddPage";
import { DashboardPage } from "../pages/DashboardPage";
import { KnowledgeDetailPage } from "../pages/KnowledgeDetailPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { SettingsPage } from "../pages/SettingsPage";
import { SourceDetailPage } from "../pages/SourceDetailPage";
import { SourcesPage } from "../pages/SourcesPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "ajouter", element: <AddPage /> },
      { path: "sources", element: <SourcesPage /> },
      { path: "sources/:sourceId", element: <SourceDetailPage /> },
      { path: "connaissances/:nodeId", element: <KnowledgeDetailPage /> },
      { path: "parametres", element: <SettingsPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

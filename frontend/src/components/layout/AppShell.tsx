import { NavLink, Outlet } from "react-router-dom";

function DashboardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 4h6v6H4V4Zm10 0h6v10h-6V4ZM4 14h6v6H4v-6Zm10 4h6v2h-6v-2Z" />
    </svg>
  );
}

function AddIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M11 5a1 1 0 0 1 2 0v6h6a1 1 0 1 1 0 2h-6v6a1 1 0 1 1-2 0v-6H5a1 1 0 1 1 0-2h6V5Z" />
    </svg>
  );
}

function SourcesIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 3a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h1v1a1 1 0 0 0 1.55.83L10 19h9a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5Zm2 5a1 1 0 0 1 1-1h8a1 1 0 1 1 0 2H8a1 1 0 0 1-1-1Zm1 3h8a1 1 0 1 1 0 2H8a1 1 0 1 1 0-2Zm-1 5a1 1 0 0 1 1-1h5a1 1 0 1 1 0 2H8a1 1 0 0 1-1-1Z" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 6a2 2 0 1 1 0-4 2 2 0 0 1 0 4Zm9-3h-1.35a7.8 7.8 0 0 0-.73-1.77l.96-.96a1 1 0 0 0 0-1.41l-1.74-1.74a1 1 0 0 0-1.41 0l-.96.96A7.8 7.8 0 0 0 14 4.35V3a1 1 0 0 0-1-1h-2a1 1 0 0 0-1 1v1.35a7.8 7.8 0 0 0-1.77.73l-.96-.96a1 1 0 0 0-1.41 0L4.12 5.86a1 1 0 0 0 0 1.41l.96.96A7.8 7.8 0 0 0 4.35 10H3a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h1.35c.16.63.41 1.22.73 1.77l-.96.96a1 1 0 0 0 0 1.41l1.74 1.74a1 1 0 0 0 1.41 0l.96-.96c.55.32 1.14.57 1.77.73V21a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-1.35a7.8 7.8 0 0 0 1.77-.73l.96.96a1 1 0 0 0 1.41 0l1.74-1.74a1 1 0 0 0 0-1.41l-.96-.96c.32-.55.57-1.14.73-1.77H21a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1Z" />
    </svg>
  );
}

export function AppShell() {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Aller au contenu
      </a>

      <aside className="sidebar">
        <NavLink className="brand" to="/" aria-label="Second Brain — accueil">
          <span className="brand-mark" aria-hidden="true">
            SB
          </span>
          <span>
            <strong>Second Brain</strong>
            <small>Sources locales</small>
          </span>
        </NavLink>

        <nav className="main-navigation" aria-label="Navigation principale">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `navigation-link${isActive ? " is-active" : ""}`
            }
          >
            <DashboardIcon />
            <span>Dashboard</span>
          </NavLink>
          <NavLink
            to="/sources"
            className={({ isActive }) =>
              `navigation-link${isActive ? " is-active" : ""}`
            }
          >
            <SourcesIcon />
            <span>Sources</span>
          </NavLink>
          <NavLink
            to="/ajouter"
            className={({ isActive }) =>
              `navigation-link${isActive ? " is-active" : ""}`
            }
          >
            <AddIcon />
            <span>Ajouter</span>
          </NavLink>
          <NavLink
            to="/parametres"
            className={({ isActive }) =>
              `navigation-link${isActive ? " is-active" : ""}`
            }
          >
            <SettingsIcon />
            <span>Paramètres</span>
          </NavLink>
        </nav>

        <p className="local-note">Les données restent sur cet ordinateur.</p>
      </aside>

      <main id="main-content" className="main-content" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}

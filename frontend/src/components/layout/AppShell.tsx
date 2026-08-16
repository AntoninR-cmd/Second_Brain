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
            <small>Notes locales</small>
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
            to="/ajouter"
            className={({ isActive }) =>
              `navigation-link${isActive ? " is-active" : ""}`
            }
          >
            <AddIcon />
            <span>Ajouter</span>
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

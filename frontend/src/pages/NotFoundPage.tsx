import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <section className="page narrow-page">
      <p className="eyebrow">Erreur 404</p>
      <h1>Cette page n’existe pas</h1>
      <p className="page-introduction">
        L’adresse demandée ne correspond à aucune page de l’application.
      </p>
      <Link className="button button-primary" to="/">
        Revenir au Dashboard
      </Link>
    </section>
  );
}

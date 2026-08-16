# Second Brain — Phase 1

Application locale permettant d’ajouter des notes en texte libre, de les conserver dans SQLite et de les retrouver sur un Dashboard après un redémarrage.

Cette phase contient uniquement FastAPI, React/TypeScript/Vite, la configuration locale, SQLite, la navigation Dashboard/Ajouter et les tests backend essentiels. Elle n’inclut aucun composant Ollama, Qdrant, embeddings, import de fichiers, RAG, clustering ou graphe.

## Prérequis Windows

- Windows 10 ou 11 avec PowerShell ;
- Python 3.10.x accessible avec `python` ;
- Node.js 20.19+ ou 22.12+ (Node 22 LTS recommandé) ;
- npm, installé avec Node.js.

Les commandes ci-dessous partent de la racine du dépôt. Pour l’emplacement actuel :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain"
```

## Installation

Dans PowerShell :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Copy-Item -LiteralPath .env.example -Destination .env -ErrorAction SilentlyContinue
.\scripts\setup.ps1
```

Le changement de politique ne vaut que pour la fenêtre PowerShell courante. Le script crée `.venv`, installe le backend en mode éditable, applique la migration SQLite et installe exactement les dépendances frontend du lockfile.

Équivalent manuel :

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m alembic upgrade head
Set-Location -LiteralPath .\frontend
npm.cmd ci
Set-Location -LiteralPath ..
```

`npm.cmd` est utilisé volontairement : il fonctionne même lorsque Windows bloque le script `npm.ps1`.

## Démarrage en développement

La méthode la plus lisible utilise deux terminaux PowerShell.

Terminal 1 — backend, depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m uvicorn second_brain.main:app --reload --host 127.0.0.1 --port 8001
```

Le port `8001` est utilisé parce que Docker Desktop occupe déjà le port `8000` sur la machine cible.

Terminal 2 — frontend, depuis la racine :

```powershell
Set-Location -LiteralPath .\frontend
npm.cmd run dev -- --host 127.0.0.1
```

Ouvrir ensuite <http://127.0.0.1:5173>. La documentation interactive de l’API est disponible sur <http://127.0.0.1:8001/docs>.

Il est aussi possible de lancer les deux applications dans un seul terminal :

```powershell
.\scripts\dev.ps1
```

Dans ce mode, le backend est lancé sans rechargement automatique et ses journaux sont écrits dans `data/logs/`. `Ctrl+C` arrête Vite puis le backend.

## Vérifications

Toutes les vérifications :

```powershell
.\scripts\check.ps1
```

Commandes séparées :

```powershell
.\.venv\Scripts\python.exe -m ruff check backend
.\.venv\Scripts\python.exe -m ruff format --check backend
.\.venv\Scripts\python.exe -m pytest
Set-Location -LiteralPath .\frontend
npm.cmd run lint
npm.cmd run build
Set-Location -LiteralPath ..
```

Les tests backend utilisent leur propre fichier SQLite temporaire. Ils vérifient notamment l’accès réel à la base, la validation du texte, la création et la lecture d’une note, le Dashboard, les pragmas SQLite et la persistance après recréation complète de l’application.

## Vérifier manuellement la persistance

1. Ouvrir la page **Ajouter** et enregistrer une note.
2. Vérifier qu’elle apparaît dans les notes récentes du Dashboard.
3. Arrêter les deux serveurs avec `Ctrl+C`.
4. Relancer les commandes de démarrage ci-dessus.
5. Recharger <http://127.0.0.1:5173> : la note doit toujours être présente.

Les données sont stockées par défaut dans `%LOCALAPPDATA%\SecondBrain\data\second_brain.sqlite3`. Ce chemin est volontairement hors du dossier synchronisé `Mon Drive` : une base SQLite active ne doit pas être synchronisée fichier par fichier pendant son utilisation.

## Configuration

Copier `.env.example` vers `.env`, puis modifier uniquement les valeurs nécessaires :

| Variable | Valeur par défaut | Rôle |
|---|---|---|
| `SECOND_BRAIN_ENV` | `development` | Environnement d’exécution |
| `SECOND_BRAIN_DATA_DIR` | `%LOCALAPPDATA%/SecondBrain/data` | Répertoire des données locales |
| `SECOND_BRAIN_DATABASE_URL` | dérivée de `DATA_DIR` | URL SQLAlchemy facultative de la base |
| `SECOND_BRAIN_ALLOWED_ORIGINS` | Vite local | Origines CORS autorisées |

Les commandes Uvicorn du README fixent explicitement l’adresse et le port. Pour les modifier, adapter à la fois la commande de lancement, `scripts/dev.ps1` et la cible du proxy dans `frontend/vite.config.ts`.

## API de la Phase 1

- `GET /api/v1/system/health` : vérifie réellement la connexion SQLite ;
- `GET /api/v1/dashboard` : retourne le compteur et les cinq notes les plus récentes ;
- `POST /api/v1/sources/manual` : ajoute une note en texte libre ;
- `GET /api/v1/sources` : liste les notes ;
- `GET /api/v1/sources/{id}` : retourne le détail d’une note.

Exemple de création directe depuis PowerShell, backend démarré :

```powershell
$body = @{ title = "Ma note"; author = "Antonin"; text = "Texte libre persistant." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/v1/sources/manual" -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## Structure principale

```text
backend/                 FastAPI, modèle SQLAlchemy, migration et tests
frontend/                React, TypeScript, Vite, Dashboard et page Ajouter
scripts/setup.ps1        installation Windows reproductible
scripts/dev.ps1          lancement local des deux applications
scripts/check.ps1        lint, tests et build
data/                    journaux de développement locaux et ignorés
.env.example             configuration locale documentée
pyproject.toml           dépendances et outils Python
alembic.ini              configuration des migrations
```

Le frontend Vite et le backend restent deux processus de développement dans cette phase. Le service du build React par FastAPI appartient à une phase ultérieure.

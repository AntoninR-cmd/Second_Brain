# Second Brain — Phase 3

Application locale pour conserver des notes et des sources TXT/SRT, puis les analyser à la demande avec Ollama. Le résumé, les connaissances atomiques, leurs tags et leurs preuves précises sont persistés dans SQLite et restent disponibles après redémarrage.

Cette phase conserve toutes les fonctions des Phases 1 et 2. Elle n’inclut ni PDF/EPUB, ni embeddings, Qdrant, recherche vectorielle, RAG, clustering, UMAP ou graphe.

## Prérequis Windows

- Windows 10 ou 11 avec PowerShell ;
- Python 3.10.x accessible avec `python` ;
- Node.js 20.19+ ou 22.12+ (Node 22 LTS recommandé) ;
- npm, installé avec Node.js ;
- Ollama pour utiliser l’analyse IA (l’application reste utilisable sans lui).

Toutes les commandes partent de la racine du dépôt :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain"
```

## Installation de l’application

Dans PowerShell :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Copy-Item -LiteralPath .env.example -Destination .env -ErrorAction SilentlyContinue
.\scripts\setup.ps1
```

Le changement de politique ne vaut que pour cette fenêtre PowerShell. Le script crée `.venv`, installe le backend en mode éditable, applique toutes les migrations SQLite et installe exactement les dépendances frontend du lockfile.

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

`npm.cmd` est utilisé volontairement : il fonctionne même lorsque Windows bloque `npm.ps1`.

## Installation explicite d’Ollama

Second Brain ne télécharge jamais Ollama ni un modèle automatiquement. Pour installer Ollama avec la commande officielle Windows :

```powershell
irm https://ollama.com/install.ps1 | iex
ollama --version
```

L’installateur graphique officiel est également disponible sur <https://ollama.com/download/windows>. Fermer puis rouvrir PowerShell si la commande `ollama` n’est pas encore reconnue.

Installer ensuite, explicitement, le modèle de génération par défaut :

```powershell
ollama pull qwen3.5:4b
ollama list
```

Ollama pour Windows démarre normalement son service local avec l’application. Si nécessaire, le lancer dans un terminal dédié :

```powershell
ollama serve
```

L’API doit alors répondre sur `http://127.0.0.1:11434`. Le navigateur ne communique jamais directement avec elle : seul FastAPI appelle Ollama.

## Démarrage en développement

La méthode la plus lisible utilise deux terminaux PowerShell.

Terminal 1 — backend, depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m uvicorn second_brain.main:app --reload --host 127.0.0.1 --port 8001
```

Terminal 2 — frontend, depuis la racine :

```powershell
Set-Location -LiteralPath .\frontend
npm.cmd run dev -- --host 127.0.0.1
```

Ouvrir <http://127.0.0.1:5173>. La documentation interactive de l’API est sur <http://127.0.0.1:8001/docs>.

Il est aussi possible de lancer les deux applications dans un seul terminal :

```powershell
.\scripts\dev.ps1
```

Dans ce mode, les journaux backend sont écrits dans `data/logs/`. `Ctrl+C` arrête Vite puis le backend.

## Tester manuellement l’analyse IA

1. Vérifier dans **Paramètres** qu’Ollama et `qwen3.5:4b` sont disponibles.
2. Ajouter une note ou importer un `.txt`/`.srt` depuis **Ajouter**.
3. Ouvrir la source, puis cliquer sur **Analyser avec l’IA**.
4. L’état passe par **En attente**, puis **Analyse en cours**. La page peut être quittée pendant le traitement.
   La progression est persistée et affiche notamment `Analyse des passages : 12 / 31`, le pourcentage et la dernière activité.
5. À la fin, vérifier le résumé, les connaissances et leurs tags.
6. Ouvrir une connaissance. Pour un SRT, vérifier l’extrait original, les indices de segments et les timestamps.
7. Redémarrer backend et frontend : le résultat doit rester présent sans nouvel appel au modèle.

Les analyses sont exécutées une par une par un worker local léger. Chaque passage validé est enregistré comme checkpoint avant de passer au suivant. Après une erreur ou un redémarrage, les passages terminés sont revalidés depuis SQLite et ne sont pas renvoyés à Ollama ; le passage interrompu est repris, puis les suivants sont traités. Les connaissances et leurs preuves définitives restent reconstruites dans une transaction finale, ce qui évite les doublons et les résultats partiels incohérents.

Le job conserve son étape, le passage courant, le total, le pourcentage, un heartbeat et un diagnostic structuré. Le frontend affiche notamment le passage fautif, l’étape, la classe d’erreur et le détail de validation. Un job `running` dont la dernière activité dépasse `JOB_STALE_HEARTBEAT_SECONDS` est signalé comme stale. Au redémarrage du processus qui possédait le worker, tout job encore `running` est récupéré proprement, qu’il soit déjà stale ou non.

Une réponse JSON ou sémantiquement invalide est retentée au plus `EXTRACTION_MAX_RETRIES` fois après une première tentative. La requête corrective reçoit le champ fautif et le schéma attendu, sans recommencer le document entier. Aucune connaissance non validée n’est insérée.

Tous les appels de génération Ollama imposent `think=false` dans le client backend. Ce réglage couvre l’analyse structurée des passages, l’extraction des connaissances et les synthèses intermédiaire, hiérarchique et finale. Il n’est pas configurable : une ancienne variable `OLLAMA_THINK` présente dans `.env` est ignorée.

Le pipeline effectue un seul appel structuré par passage : cette réponse fournit simultanément le résumé intermédiaire, au plus deux connaissances, leurs tags et leur provenance. Les synthèses hiérarchique et finale restent des appels distincts après validation de tous les passages.

Chaque tentative écrit une ligne de métriques sans le prompt, la réponse, le titre ni le contenu de la source : type d’appel, durée totale en secondes, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, numéro de retry et résultat. Chaque checkpoint journalise aussi le nombre de connaissances validées pour ce passage. À la fin ou à l’échec, une ligne `Analysis benchmark` récapitule passages, appels, connaissances, durée totale et moyenne, tokens, durées Ollama et retries. Les durées fournies par Ollama restent en nanosecondes. Avec les deux terminaux, ces lignes apparaissent dans le terminal backend ; avec `scripts\dev.ps1`, elles sont dans `data\logs\backend-dev.stderr.log`.

Si Ollama est arrêté, la bibliothèque, les imports et les notes continuent de fonctionner. La page **Paramètres** affiche un état dégradé et le lancement d’une analyse renvoie une explication. Si Ollama s’arrête pendant un traitement, la source passe en erreur et peut être relancée après redémarrage d’Ollama.

### Reprendre une analyse interrompue et comparer la durée

1. Mettre la base à niveau, puis vérifier que `.env` contient les limites optimisées. Une ancienne valeur présente dans `.env` surcharge toujours le nouveau défaut :

   ```dotenv
   OLLAMA_GENERATION_MODEL=qwen3.5:4b
   OLLAMA_EXTRACTION_TEMPERATURE=0.0
   OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS=512
   OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY=512
   OLLAMA_NUM_PREDICT_FINAL_SUMMARY=1024
   EXTRACTION_MAX_RETRIES=1
   EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE=2
   JOB_STALE_HEARTBEAT_SECONDS=120
   ```

   ```powershell
   .\.venv\Scripts\python.exe -m alembic upgrade head
   ```

2. Redémarrer le backend, ouvrir la source qui avait échoué et copier son UUID depuis l’URL. Un ancien job resté `running` est repris automatiquement au démarrage. Pour un job `failed`, le script ci-dessous crée la relance. Les analyses exécutées avant la migration `0005` ne possédaient pas encore de payload de checkpoint complet : leurs anciens passages doivent être recalculés une fois. Toutes les nouvelles validations deviennent ensuite reprenables.

3. Exécuter dans un autre PowerShell :

   ```powershell
   $sourceId = "COLLER-ICI-L-UUID-DE-LA-SOURCE"
   $api = "http://127.0.0.1:8001/api/v1"
   $wallStart = Get-Date

   try {
       $job = Invoke-RestMethod -Method Get -Uri "$api/sources/$sourceId/analysis"
   } catch {
       $job = $null
   }

   if (-not $job -or $job.status -notin @("pending", "running")) {
       if ($job -and $job.status -eq "succeeded") {
           throw "Cette source est déjà analysée. Réimportez le même SRT pour un benchmark neuf."
       }
       $job = Invoke-RestMethod -Method Post -Uri "$api/sources/$sourceId/analyze"
   }

   do {
       Start-Sleep -Seconds 1
       $job = Invoke-RestMethod -Method Get -Uri "$api/jobs/$($job.id)"
       Write-Host ("{0,3}%  {1}" -f $job.progress_percent, $job.progress_message)
   } while ($job.status -in @("pending", "running"))

   $wallSeconds = [Math]::Round(((Get-Date) - $wallStart).TotalSeconds, 2)
   Write-Host "Statut final : $($job.status)"
   Write-Host "Durée totale observée : $wallSeconds secondes"

   if ($job.started_at -and $job.finished_at) {
       $startedAt = [DateTimeOffset]::Parse($job.started_at)
       $finishedAt = [DateTimeOffset]::Parse($job.finished_at)
       $workerSeconds = [Math]::Round(($finishedAt - $startedAt).TotalSeconds, 2)
       Write-Host "Durée du worker : $workerSeconds secondes"
   }

   Write-Host "SourcePassages : $($job.progress_total)"
   Write-Host "Appels Ollama : $($job.llm_call_count)"
   Write-Host "Retries : $($job.llm_retry_count)"
   Write-Host "KnowledgeNodes : $($job.knowledge_node_count)"
   Write-Host "Tokens entrée : $($job.prompt_eval_count)"
   Write-Host "Tokens sortie : $($job.eval_count)"
   ```

Comparer `Durée du worker` à la mesure précédente. Pour une comparaison équitable, garder le même fichier, le même modèle et les mêmes limites de chunks, et indiquer dans les deux mesures si le modèle était déjà chargé en mémoire. Une source déjà analysée doit être réimportée pour un benchmark neuf ; une source en erreur peut être relancée directement et réutilise ses checkpoints valides.

## Vérifications automatiques

Toutes les vérifications :

```powershell
.\scripts\check.ps1
```

Commandes séparées :

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend
.\.venv\Scripts\python.exe -m ruff format --check backend
Set-Location -LiteralPath .\frontend
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run build
Set-Location -LiteralPath ..
```

La suite normale utilise un faux client Ollama et ne requiert ni service ni modèle installé. Elle couvre aussi les migrations, la reprise des jobs, le chunking TXT/SRT, les sorties JSON strictes, les erreurs Ollama, la provenance et toutes les fonctions des Phases 1 et 2.

## Persistance locale

La base est stockée par défaut dans `%LOCALAPPDATA%\SecondBrain\data\second_brain.sqlite3`. Chaque fichier importé est copié sans modification sous `%LOCALAPPDATA%\SecondBrain\data\originals\<UUID>\original.<extension>`.

Cet emplacement est volontairement hors du dossier synchronisé `Mon Drive` : une base SQLite active ne doit pas être synchronisée fichier par fichier pendant son utilisation. Les données d’exécution, les bases SQLite et les répertoires `originals/` sont ignorés par Git.

## Configuration

Copier `.env.example` vers `.env`, puis redémarrer le backend après toute modification :

| Variable | Défaut | Rôle |
|---|---:|---|
| `SECOND_BRAIN_DATA_DIR` | `%LOCALAPPDATA%/SecondBrain/data` | Données locales |
| `SECOND_BRAIN_DATABASE_URL` | dérivée de `DATA_DIR` | URL SQLAlchemy facultative |
| `SECOND_BRAIN_ALLOWED_ORIGINS` | Vite local | Origines CORS |
| `SECOND_BRAIN_MAX_UPLOAD_MB` | `20` | Limite TXT/SRT en Mio |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | API Ollama locale |
| `OLLAMA_GENERATION_MODEL` | `qwen3.5:4b` | Modèle de génération |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `600` | Timeout d’une génération |
| `OLLAMA_READINESS_TIMEOUT_SECONDS` | `5` | Timeout du diagnostic |
| `OLLAMA_NUM_CTX` | `8192` | Fenêtre de contexte demandée |
| `OLLAMA_TEMPERATURE` | `0.2` | Température des synthèses |
| `OLLAMA_EXTRACTION_TEMPERATURE` | `0.0` | Température déterministe de l’analyse structurée des passages |
| `OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS` | `512` | Plafond de sortie pour résumé intermédiaire et connaissances d’un passage |
| `OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY` | `512` | Plafond de sortie d’une synthèse hiérarchique intermédiaire |
| `OLLAMA_NUM_PREDICT_FINAL_SUMMARY` | `1024` | Plafond de sortie du résumé final |
| `OLLAMA_KEEP_ALIVE` | `5m` | Durée de maintien du modèle en mémoire |
| `CHUNK_TARGET_TOKENS` | `800` | Taille cible des passages |
| `CHUNK_MAX_TOKENS` | `1200` | Limite estimée d’un passage |
| `CHUNK_OVERLAP_SEGMENTS` | `2` | Chevauchement des passages SRT |
| `CHUNK_SRT_PAUSE_MS` | `2500` | Pause favorisant une coupure SRT |
| `EXTRACTION_MAX_RETRIES` | `1` | Nouvelle tentative maximale après sortie JSON/sémantique invalide |
| `EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE` | `2` | Maximum de connaissances utiles demandé par passage |
| `JOB_STALE_HEARTBEAT_SECONDS` | `120` | Délai sans heartbeat avant de signaler un job `running` comme stale |

Les limites de chunks sont des estimations indépendantes du modèle. Une entrée SRT reste entière tant qu’elle tient sous le plafond ; seule une entrée exceptionnellement longue utilise un découpage de secours borné. Les textes suivent d’abord les paragraphes et les phrases.

## API de la Phase 3

- `GET /api/v1/system/health` : connexion SQLite ;
- `GET /api/v1/system/readiness` : état Ollama et disponibilité du modèle ;
- `POST /api/v1/sources/manual` : note libre ;
- `POST /api/v1/sources/upload` : import TXT/SRT ;
- `GET /api/v1/sources` et `GET /api/v1/sources/{id}` : liste et détail ;
- `GET /api/v1/sources/{id}/segments` : segments SRT ;
- `POST /api/v1/sources/{id}/analyze` : mise en file de l’analyse ;
- `GET /api/v1/sources/{id}/analysis` : dernier job et progression persistante de la source ;
- `GET /api/v1/jobs/{id}` : état détaillé du traitement ;
- `GET /api/v1/sources/{id}/nodes` : connaissances d’une source ;
- `GET /api/v1/nodes/{id}` : connaissance, source et preuves exactes.

## Structure principale

```text
backend/alembic/                    migrations SQLite
backend/src/second_brain/llm/      client Ollama, schémas JSON et prompts
backend/src/second_brain/pipeline/ chunking indépendant du modèle
backend/src/second_brain/jobs/     worker local persistant
backend/src/second_brain/services/ pipeline d’analyse hiérarchique
frontend/                           React, Sources, analyse, connaissances, Paramètres
scripts/                            installation, démarrage et vérifications Windows
.env.example                        configuration documentée
pyproject.toml                      dépendances et outils Python
```

Le traitement IA reste entièrement local et déclenché manuellement. Aucun composant de la Phase 4 n’est présent.

# Second Brain — Phase 6B

Application locale pour conserver des notes et des sources TXT/SRT, les analyser à la demande avec Ollama, retrouver les connaissances atomiques par similarité sémantique, obtenir une réponse sourcée et explorer le modèle mathématique versionné du cerveau dans une carte interactive. Les sources, résumés, connaissances, tags, preuves et états d’indexation sont persistés dans SQLite et restent disponibles après redémarrage.

Cette phase conserve toutes les fonctions des Phases 1 à 6A. Elle visualise avec Sigma.js et Graphology les relations sémantiques, clusters hiérarchiques et coordonnées 2D déjà calculés par la Phase 6A. Elle ne recalcule ni similarité, ni clustering, ni UMAP dans le navigateur et n’ajoute aucune fonctionnalité de Phase 7.

## Prérequis Windows

- Windows 10 ou 11 avec PowerShell ;
- Python 3.10.x accessible avec `python` ;
- Node.js 20.19+ ou 22.12+ (Node 22 LTS recommandé) ;
- npm, installé avec Node.js ;
- Ollama pour utiliser l’analyse IA, l’indexation, la recherche sémantique et les réponses RAG (les sources, connaissances et un cerveau déjà construit restent consultables sans lui).

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

Le changement de politique ne vaut que pour cette fenêtre PowerShell. Le script crée `.venv`, installe le backend en mode éditable, applique toutes les migrations SQLite et installe exactement les dépendances frontend du lockfile. Les bibliothèques mathématiques de la Phase 6A (`numpy`, `scikit-learn` et `umap-learn`) sont installées par cette même commande : aucune installation Windows séparée n’est nécessaire.

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

Installer ensuite, explicitement, les deux modèles par défaut :

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
ollama list
```

`qwen3.5:4b` produit les analyses de Phase 3, les réponses RAG et, lorsqu’il est disponible, des labels courts pour les clusters. `qwen3-embedding:0.6b` encode les connaissances et les requêtes de recherche ; il ne génère aucun texte. Second Brain ne lance jamais `ollama pull` à votre place. La construction mathématique peut utiliser des labels déterministes si le modèle de génération est indisponible.

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

Si Ollama est arrêté, la bibliothèque, les imports et les notes continuent de fonctionner. La page **Paramètres** affiche un état dégradé ; une analyse ou une question RAG renvoie une explication exploitable. Si Ollama s’arrête pendant un traitement, la source passe en erreur et peut être relancée après redémarrage d’Ollama.

## Indexer et rechercher les connaissances

La page **Paramètres** présente séparément le modèle de génération et le modèle d’embedding. La section **Index vectoriel local** indique :

- le nombre total de `KnowledgeNodes` dans SQLite ;
- les connaissances correctement indexées ;
- les connaissances nouvelles, modifiées ou en erreur ;
- les éventuels points Qdrant orphelins ;
- le modèle, son digest lorsqu’Ollama le fournit, la dimension observée, la distance cosinus et la génération logique du profil actif ;
- le job courant, sa progression persistante et sa dernière activité.

Procédure depuis l’interface :

1. Vérifier dans **Paramètres** qu’Ollama et `qwen3-embedding:0.6b` sont disponibles.
2. Cliquer sur **Indexer les connaissances**. L’opération traite uniquement les connaissances non indexées, obsolètes ou en erreur et nettoie les points orphelins.
3. Suivre la progression, par exemple `17 / 39 connaissances`. La page peut être quittée pendant le traitement.
4. Ouvrir **Recherche IA**, saisir une question, choisir le mode, puis cliquer sur **Envoyer**.
5. Examiner la réponse, les connaissances récupérées et celles réellement citées.
6. Ouvrir une connaissance pour retrouver son contenu complet, ses tags, sa source et ses preuves exactes.

Le retrieval compare toujours les vecteurs sans seuil arbitraire restrictif. L’API sémantique Phase 4 reste disponible seule ; l’interface **Recherche IA** transmet par défaut les huit meilleurs résultats au pipeline RAG, dans la limite du budget de contexte.

### Architecture de l’index local

SQLite reste la source de vérité. Qdrant fonctionne en mode local persistant, dans le processus backend, sans serveur, Docker ou compte cloud. Son répertoire par défaut est `%LOCALAPPDATA%\SecondBrain\data\qdrant` et peut être entièrement reconstruit à partir de SQLite.

Pour chaque connaissance, le texte sémantique est exactement `titre + ligne vide + contenu`. Les UUID, chemins, dates, timestamps, tags et preuves complètes sont volontairement exclus afin de mesurer la similarité du contenu lui-même. Un fingerprint SHA-256 de ce texte et le profil d’embedding permettent de détecter une création ou une modification.

Les connaissances sont encodées par lots, puis chaque point Qdrant conserve seulement l’UUID du `KnowledgeNode`, son `source_id` et le fingerprint nécessaire à la reprise. Après une recherche, FastAPI recharge les connaissances, tags, sources et preuves depuis SQLite. Une panne d’Ollama ou de Qdrant ne supprime et ne modifie jamais les données métier.

L’indexation est idempotente et reprenable. Chaque lot validé est d’abord écrit dans Qdrant, puis son checkpoint est persisté dans SQLite. Si le processus s’arrête entre ces deux opérations, la reprise reconnaît le point déjà valide et recrée le checkpoint sans recalculer son embedding. Les KnowledgeNodes créés après le début d’un job seront pris en charge par le job suivant.

Le changement de `OLLAMA_EMBEDDING_MODEL`, de digest ou de dimension rend le profil actif incompatible au lieu de mélanger deux espaces vectoriels. Utiliser alors **Reconstruire l’index** : après confirmation, une nouvelle collection versionnée est construite, sa dimension réelle est observée, puis elle devient active uniquement lorsque la reconstruction est complète. L’ancien index reste une donnée dérivée ; les connaissances SQLite ne sont jamais touchées.

### Commandes API équivalentes sous Windows

Avec le backend démarré sur le port `8001` :

```powershell
$api = "http://127.0.0.1:8001/api/v1"

# État des modèles et de l’index
$status = Invoke-RestMethod -Method Get -Uri "$api/vector-index/status"
$status | Format-List

# Indexation incrémentale
$job = Invoke-RestMethod -Method Post -Uri "$api/vector-index/index"
do {
    Start-Sleep -Seconds 1
    $job = Invoke-RestMethod -Method Get -Uri "$api/vector-index/jobs/$($job.id)"
    Write-Host ("{0,3}%  {1}" -f $job.progress_percent, $job.progress_message)
} while ($job.status -in @("pending", "running"))

if ($job.status -ne "succeeded") {
    $errorText = $job.error_detail
    if (-not $errorText) { $errorText = $job.error_message }
    throw "Indexation interrompue : $errorText"
}

# Recherche sémantique Phase 4, sans génération de réponse
$searchBody = @{
    query = "Comment mieux récupérer entre deux séances difficiles ?"
    top_k = 5
} | ConvertTo-Json

$results = Invoke-RestMethod `
    -Method Post `
    -Uri "$api/search/semantic" `
    -ContentType "application/json" `
    -Body $searchBody

$results.items | ForEach-Object {
    [PSCustomObject]@{
        score = [Math]::Round($_.score, 4)
        titre = $_.knowledge_node.title
        source = $_.source.title
        lien = "http://127.0.0.1:5173$($_.href)"
    }
} | Format-Table -AutoSize
```

La reconstruction complète est volontairement explicite :

```powershell
$api = "http://127.0.0.1:8001/api/v1"
$body = @{ confirm = $true } | ConvertTo-Json
$job = Invoke-RestMethod `
    -Method Post `
    -Uri "$api/vector-index/rebuild" `
    -ContentType "application/json" `
    -Body $body
$job
```

Dans l’interface, cette même action demande une confirmation avant de démarrer.

## Poser une question avec le RAG sourcé

Chaque question est indépendante : cette phase ne crée ni conversation ni historique persistant. Le backend exécute exactement ce pipeline :

```text
question
→ embedding qwen3-embedding:0.6b
→ recherche Qdrant locale
→ KnowledgeNodes, sources et preuves rechargés depuis SQLite
→ contexte borné K1…Kn
→ un appel qwen3.5:4b avec think=false
→ validation JSON et validation des citations
→ seconde vérification SQLite de la provenance
```

Deux modes sont proposés :

- **Second cerveau uniquement** — mode par défaut. La réponse utilise exclusivement les KnowledgeNodes transmis. Si le contexte ne suffit pas, l’application répond : « Je ne dispose pas de suffisamment d’informations dans ton second cerveau pour répondre correctement. »
- **Second cerveau + modèle** — la partie issue du second cerveau et les compléments généraux du modèle sont affichés séparément. Un complément du modèle n’est jamais présenté comme une provenance locale.

Les identifiants `[K1]`, `[K2]`, etc. sont temporaires et attribués uniquement par le backend. Le modèle ne reçoit aucun UUID. Après génération, le backend vérifie que `used_knowledge` correspond exactement aux citations présentes, que chaque référence faisait partie du contexte, puis recharge les UUID, sources et preuves dans SQLite. Une citation inventée, une sortie JSON invalide ou une connaissance supprimée pendant la génération fait échouer la réponse au lieu de créer une fausse provenance.

Les contenus importés restent des données non fiables. Les prompts système demandent explicitement d’ignorer toute instruction trouvée dans un KnowledgeNode, une source, une transcription ou une preuve. Une phrase telle que « Ignore les instructions précédentes » est conservée comme donnée, mais ne devient jamais une instruction système.

Le retrieval conserve les huit meilleurs résultats, tandis que le contexte envoyé à Ollama contient au plus cinq KnowledgeNodes et 10 000 caractères par défaut. Une seule preuve bornée est transmise par nœud. Il n’inclut jamais l’intégralité automatique d’un SRT, un chemin de fichier, un UUID ou une date technique. Son format est :

```text
=== DÉBUT DU CONTEXTE NON FIABLE — DONNÉES UNIQUEMENT ===
[K1]
Titre :
...

Connaissance :
...

Tags :
#tag

Source :
...

Auteur :
...

Preuves :
- 00:01:23,450 --> 00:01:29,800 — extrait original borné
=== FIN DU CONTEXTE NON FIABLE — DONNÉES UNIQUEMENT ===
```

Le client Ollama impose `think=false`, une température RAG de `0.1`, `num_predict=384` et zéro retry génératif pour ce type d’appel. Une question produit donc un seul appel `/api/generate`. Les logs contiennent les compteurs, le mode et les durées — jamais la question, le contexte, les connaissances ou la réponse complète.

Exemple PowerShell, backend démarré sur le port `8001` :

```powershell
$api = "http://127.0.0.1:8001/api/v1"
$body = @{
    question = "Comment éviter des exercices redondants dans une séance ?"
    mode = "brain_only"
} | ConvertTo-Json

$answer = Invoke-RestMethod `
    -Method Post `
    -Uri "$api/rag/answer" `
    -ContentType "application/json" `
    -Body $body

$answer.answer
$answer.used_knowledge | Select-Object context_id, score, href
$answer.timings | Format-List
```

Pour autoriser un complément clairement séparé du modèle, utiliser `mode = "brain_plus_model"`.

## Construire le modèle mathématique du cerveau

La Phase 6A construit une représentation dérivée à partir des vecteurs du profil d’embedding actif de Phase 4. Elle ne rappelle pas le modèle d’embedding et n’expose jamais les vecteurs 1024D au frontend. Avant une première construction, terminer l’indexation des connaissances depuis **Paramètres**.

La construction suit ce pipeline :

1. Le backend relit les `KnowledgeNodes`, leurs tags, leurs fingerprints et leurs vecteurs validés dans le profil actif.
2. Un graphe k-nearest-neighbors borné conserve au plus quelques relations par connaissance, sans persister le graphe complet de toutes les paires. Le score d’une relation est `0,95 × similarité cosinus + 0,05 × Jaccard(tags)`. Les embeddings restent donc dominants et le calcul fonctionne sans tags.
3. Une PCA déterministe réduit uniquement la dimension de travail, puis un clustering hiérarchique agglomératif sélectionne des partitions selon leur silhouette. Les connaissances trop isolées peuvent rester non assignées au lieu d’être forcées dans un thème.
4. UMAP calcule ensuite les seules coordonnées 2D, avec une graine fixe, puis les normalise. Ces coordonnées ne servent jamais d’entrée au clustering.
5. Des labels déterministes issus des tags et titres sont toujours disponibles. Lorsque `qwen3.5:4b` répond, plusieurs clusters sont nommés dans un même appel structuré, avec `think=false`, une température faible et une sortie courte. Le labeling peut être relancé séparément.

La hiérarchie s’adapte au corpus : racine **Second Brain**, grands domaines, thèmes lorsque les données justifient ce niveau, puis KnowledgeNodes. La position d’un cluster est dérivée des positions de ses membres. Une profondeur ou une appartenance incohérente n’est pas imposée aux petits corpus.

Chaque construction produit un `BrainProfile` versionné qui enregistre le profil d’embedding, le digest du modèle, les algorithmes, les paramètres, le fingerprint d’entrée, les statistiques et les durées. Une modification des connaissances, de leurs embeddings ou des paramètres rend ce profil `stale`. La reconstruction suivante est préparée à côté de l’ancienne version ; la version précédemment valide reste lisible tant que la nouvelle n’est pas terminée.

`BrainProfile`, `BrainCluster`, `BrainNodeLayout` et `BrainEdge` sont persistés dans SQLite pour éviter de recalculer le layout à chaque lecture. Ils restent des données dérivées : leur suppression, leur corruption ou un échec de reconstruction ne modifie jamais les sources, connaissances, preuves, embeddings Qdrant ou réponses RAG. Le cerveau peut être entièrement reconstruit depuis les données validées des Phases 1 à 4.

La page **Paramètres** affiche l’état `ready`, `stale`, `building` ou `error`, la progression persistante, les compteurs et les durées. Elle propose **Construire/Recalculer le cerveau** et **Relancer les labels**. La carte de la Phase 6B lit ce profil sans déclencher de reconstruction.

Commandes API équivalentes, backend démarré sur le port `8001` :

```powershell
$api = "http://127.0.0.1:8001/api/v1"

# Profil actif, fraîcheur et dernier job
$brain = Invoke-RestMethod -Method Get -Uri "$api/brain/status"
$brain | Format-List

# Reconstruction explicite et progression persistante
$body = @{ confirm = $true } | ConvertTo-Json
$job = Invoke-RestMethod `
    -Method Post `
    -Uri "$api/brain/rebuild" `
    -ContentType "application/json" `
    -Body $body

do {
    Start-Sleep -Seconds 1
    $job = Invoke-RestMethod -Method Get -Uri "$api/brain/jobs/$($job.id)"
    Write-Host ("{0,3}%  {1}" -f $job.progress_percent, $job.progress_message)
} while ($job.status -in @("pending", "running"))

if ($job.status -ne "succeeded") {
    throw "Construction interrompue : $($job.error_detail)"
}

# Données hiérarchiques et graphe parcimonieux, sans vecteurs bruts
$domains = Invoke-RestMethod -Method Get -Uri "$api/brain/clusters?level=1"
$graph = Invoke-RestMethod -Method Get -Uri "$api/brain/graph?level=1"

# Relabeling facultatif, sans recalculer le clustering ni UMAP
$labelJob = Invoke-RestMethod `
    -Method Post `
    -Uri "$api/brain/relabel" `
    -ContentType "application/json" `
    -Body $body
```

## Explorer la carte interactive du cerveau

La route <http://127.0.0.1:5173/cerveau> affiche le profil valide avec **Sigma.js `3.0.3`** et **Graphology `0.26.0`**, versions épinglées dans `frontend/package.json` et `frontend/package-lock.json`. L’intégration utilise directement Sigma plutôt que `@react-sigma/core` : le composant de canevas possède ainsi explicitement l’unique renderer WebGL, ses écouteurs et sa destruction, sans ajouter une couche React dont la compatibilité devrait être suivie séparément. La branche alpha/bêta de Sigma v4 n’est pas utilisée.

La carte n’appelle jamais Ollama. Elle utilise seulement le profil mathématique déjà construit, ses coordonnées UMAP et les endpoints `/api/v1/brain/*`. Elle reste donc navigable lorsque le service Ollama est arrêté.

### Architecture frontend

Le chargement et le rendu restent séparés :

```text
client API /brain/*
→ contrats TypeScript Brain
→ conversion isolée en graphe Graphology
→ renderer Sigma stable pendant le rendu React
→ état d’interaction conservé dans React
```

- Le client API charge le statut, la hiérarchie, le niveau ou cluster courant et les détails nécessaires.
- Le module Graphology transforme les nœuds et arêtes déjà calculés par la Phase 6A. Il valide les coordonnées, ignore les arêtes invalides et ne recalcule aucune similarité.
- Sigma consomme directement les positions `x/y` persistées. Aucun ForceAtlas2 ou nouveau layout n’est lancé dans le navigateur.
- React conserve le cluster courant, le fil d’Ariane, le zoom sémantique, le hover, la sélection, la recherche et l’ouverture du panneau. Ces états ne sont pas détournés vers Graphology.
- Les couleurs sont dérivées de façon déterministe de l’UUID de la famille de domaine. Les thèmes héritent de sa teinte avec une variation stable ; l’information reste aussi portée par les labels, tailles et états visuels.

L’instance Sigma et le graphe Graphology ne sont pas recréés à chaque render. Les événements caméra, souris et tactiles sont retirés et le renderer WebGL est détruit lors du démontage du canevas.

### Zoom sémantique et navigation

La carte charge des données adaptées à la profondeur courante, plutôt que de superposer toute la hiérarchie :

1. La vue globale demande `/brain/graph?level=1` et montre principalement les grands domaines ainsi que leurs relations agrégées.
2. Un clic sélectionne un domaine ou thème et affiche son effectif et ses enfants. **Entrer dans ce thème**, le double clic ou un zoom rapproché charge `/brain/graph?cluster_id=<UUID>`.
3. Tant que le cluster possède des enfants, la carte présente ces sous-thèmes et leurs relations agrégées.
4. Dans un thème feuille, elle présente les KnowledgeNodes à leurs coordonnées Phase 6A et les `BrainEdges` locales.
5. Le fil d’Ariane **Second Brain > domaine > thème** recharge le niveau choisi sans perdre la compréhension du chemin parcouru.

Les seuils de caméra distinguent les vues domaines, thèmes et connaissances. Un payload feuille reste visible même si son niveau ne contient que des KnowledgeNodes. Les déplacements issus d’un clic, d’une recherche ou du fil d’Ariane animent la caméra ; les contrôles permettent aussi zoom avant, zoom arrière, recentrage, retour global et plein écran.

### Labels, relations et sélection

- À l’échelle globale, les labels de domaines sont prioritaires.
- Au niveau intermédiaire, un nombre borné de labels de thèmes est conservé.
- Au niveau feuille, seuls les principaux labels de connaissances sont forcés ; le hover, la sélection et un résultat de recherche restent toujours étiquetés.
- Les arêtes agrégées sont utilisées pour les clusters. Les arêtes individuelles ne sont chargées qu’au niveau des connaissances.
- Sans sélection, les liens les plus faibles sont progressivement masqués selon la distribution des scores du graphe visible, sans ajouter de seuil sémantique métier.
- Avec un KnowledgeNode sélectionné, ses voisins directs et ses arêtes sont renforcés, tandis que les éléments sans relation directe sont atténués.

La taille d’un cluster croît modérément avec son effectif. Les KnowledgeNodes utilisent une taille presque uniforme afin de ne pas suggérer une importance métier qui n’existe pas.

### Recherche et panneau de détail

Le champ de recherche appelle `GET /api/v1/brain/search?q=...`. Cette recherche textuelle locale couvre les titres des KnowledgeNodes et les labels de clusters du profil actif ; elle n’effectue aucun embedding et ne dépend pas d’Ollama. Choisir un résultat ouvre ses ancêtres, charge le bon niveau, anime la caméra vers ses coordonnées et le met temporairement en évidence.

Le clic sur un cluster ouvre son descriptif, son effectif, ses sous-thèmes et l’action **Entrer dans ce thème**. Le clic sur une connaissance charge son détail métier depuis SQLite et ouvre un panneau latéral contenant titre, contenu, tags, source, auteur, preuves et localisateurs SRT. **Ouvrir la connaissance** mène à la page existante `/connaissances/<UUID>` ; cette page demeure la seule vue détaillée complète. Sur une largeur mobile, le panneau devient un drawer superposé et le mode plein écran laisse le maximum d’espace au graphe.

### États de disponibilité

- `ready` : le profil actif est affiché normalement ;
- `stale` : l’ancien profil reste consultable avec l’avertissement « Le cerveau ne contient pas encore les dernières connaissances. » ;
- `building` : si un ancien profil valide existe, il reste visible pendant la reconstruction ; sinon l’écran affiche la progression ;
- `error` : une version valide antérieure reste consultable lorsqu’elle existe, sinon le diagnostic renvoie vers **Paramètres** ;
- sans profil : la carte explique qu’il faut d’abord indexer les connaissances et construire le cerveau dans **Paramètres**.

### Installation et commandes frontend sous Windows

Installation complète recommandée depuis la racine :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Copy-Item -LiteralPath .env.example -Destination .env -ErrorAction SilentlyContinue
.\scripts\setup.ps1
```

Le script exécute déjà `npm.cmd ci`. Pour réinstaller uniquement le frontend à partir du lockfile :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain\frontend"
npm.cmd ci
```

La commande qui verrouille explicitement les deux bibliothèques du graphe lors d’une mise à jour de dépendances est :

```powershell
npm.cmd install --save-exact sigma@3.0.3 graphology@0.26.0
```

Elle n’est pas nécessaire après `setup.ps1` ou `npm.cmd ci` lorsque le lockfile du dépôt est déjà présent.

Démarrage backend et frontend :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain"
.\scripts\dev.ps1
```

Pour démarrer uniquement Vite après avoir lancé FastAPI séparément :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain\frontend"
npm.cmd run dev -- --host 127.0.0.1
```

Tests et build frontend seuls :

```powershell
Set-Location -LiteralPath "C:\Users\Antonin\Mon Drive\Second Brain\Second_Brain\frontend"
npm.cmd run test
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run build
```

### Validation manuelle du profil réel

1. Démarrer l’application, vérifier dans **Paramètres** que le cerveau possède un profil valide, puis ouvrir <http://127.0.0.1:5173/cerveau>.
2. En vue globale, vérifier que **Principes d’entraînement en musculation** et **Fondements du style vestimentaire** apparaissent clairement, sans KnowledgeNodes ni arêtes individuelles superposés.
3. Sélectionner puis entrer dans **Principes d’entraînement en musculation**. Vérifier les quatre thèmes **Organisation des séances d’entraînement**, **Ciblage musculaire et anatomie**, **Gestion du volume et de l’intensité** et **Programmation périodisée en sport**.
4. Revenir à **Second Brain** avec le fil d’Ariane, entrer dans **Fondements du style vestimentaire**, puis vérifier **Harmonie entre corps et tenue** et **Techniques de colorimodulation**.
5. Entrer dans un thème feuille : vérifier les KnowledgeNodes, leur hover, leur sélection, les voisins et arêtes renforcés, puis le contenu, la source, les preuves et les timestamps éventuels dans le panneau.
6. Cliquer sur **Ouvrir la connaissance**, puis utiliser **Revenir au cerveau**.
7. Rechercher le titre d’une connaissance et le nom d’un cluster. Vérifier le chargement de leurs ancêtres, l’animation de caméra et la mise en évidence.
8. Tester zoom `+`/`−`, recentrage, retour global, plein écran et navigation par fil d’Ariane. Réduire ensuite la fenêtre à une largeur de téléphone et vérifier le drawer de détail.
9. Arrêter Ollama sans arrêter FastAPI, recharger `/cerveau` et vérifier que le profil valide reste navigable.

Le panneau **Performances** de la barre d’outils expose les mesures locales API, construction Graphology, première peinture Sigma, navigation et hover. Relever ces valeurs sur la machine testée ; le README ne fixe pas de résultat matériel non mesuré.

### Diagnostic des erreurs vectorielles

- **Ollama indisponible** : démarrer Ollama, puis actualiser **Paramètres**. SQLite reste accessible.
- **Modèle absent** : exécuter `ollama pull qwen3-embedding:0.6b`, ou installer le modèle configuré dans `.env`.
- **Index incompatible** : le modèle configuré ou la dimension produite diffère du profil actif ; lancer une reconstruction complète.
- **Index indisponible ou endommagé** : arrêter les autres processus backend qui pourraient verrouiller le même Qdrant local, puis reconstruire l’index dérivé. Ne pas supprimer la base SQLite.
- **Job interrompu/stale** : redémarrer le backend. Le worker local récupère le job persistant et réutilise les lots déjà validés.

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
.\.venv\Scripts\python.exe -m alembic check
Set-Location -LiteralPath .\frontend
npm.cmd run test
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run build
Set-Location -LiteralPath ..
git diff --check
```

La suite normale utilise de faux clients Ollama pour la génération et les embeddings ; elle ne requiert aucun modèle installé. Les tests Qdrant utilisent un répertoire local temporaire. Ils couvrent notamment les migrations, l’indexation et sa reprise, la recherche enrichie depuis SQLite, le contexte RAG borné, les deux modes, les sorties JSON strictes, les citations inventées, les suppressions concurrentes, la protection contre les instructions contenues dans les sources, le graphe kNN, la hiérarchie, le layout reproductible et la reconstruction versionnée, ainsi que toutes les fonctions des Phases 1 à 6A.

Les tests frontend de la Phase 6B ciblent les contrats API, la conversion Graphology, le zoom sémantique, les règles de visibilité des labels et arêtes, la recherche, la sélection, le fil d’Ariane, le panneau de détail et les états de disponibilité. Ils privilégient les modules purs et les interactions DOM sans tenter de valider profondément WebGL dans jsdom ; aucun test automatisé ne dépend d’Ollama.

## Persistance locale

La base est stockée par défaut dans `%LOCALAPPDATA%\SecondBrain\data\second_brain.sqlite3`. Chaque fichier importé est copié sans modification sous `%LOCALAPPDATA%\SecondBrain\data\originals\<UUID>\original.<extension>`. L’index Qdrant dérivé est conservé sous `%LOCALAPPDATA%\SecondBrain\data\qdrant`. Les profils, clusters, positions et arêtes du cerveau sont des tables dérivées de la même base SQLite et restent entièrement reconstructibles.

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
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Modèle distinct pour indexation et recherche |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `600` | Timeout d’une génération |
| `OLLAMA_EMBEDDING_TIMEOUT_SECONDS` | `600` | Timeout d’un lot d’embeddings |
| `OLLAMA_READINESS_TIMEOUT_SECONDS` | `5` | Timeout du diagnostic |
| `OLLAMA_NUM_CTX` | `8192` | Fenêtre de contexte demandée |
| `OLLAMA_TEMPERATURE` | `0.2` | Température des synthèses |
| `OLLAMA_EXTRACTION_TEMPERATURE` | `0.0` | Température déterministe de l’analyse structurée des passages |
| `OLLAMA_RAG_TEMPERATURE` | `0.1` | Température faible de la réponse RAG |
| `OLLAMA_CLUSTER_LABEL_TEMPERATURE` | `0.0` | Température déterministe du labeling de clusters |
| `OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS` | `512` | Plafond de sortie pour résumé intermédiaire et connaissances d’un passage |
| `OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY` | `512` | Plafond de sortie d’une synthèse hiérarchique intermédiaire |
| `OLLAMA_NUM_PREDICT_FINAL_SUMMARY` | `1024` | Plafond de sortie du résumé final |
| `OLLAMA_NUM_PREDICT_RAG` | `384` | Plafond de sortie d’une réponse RAG |
| `OLLAMA_NUM_PREDICT_CLUSTER_LABELS` | `384` | Plafond d’un lot de labels de clusters |
| `OLLAMA_KEEP_ALIVE` | `5m` | Durée de maintien du modèle en mémoire |
| `CHUNK_TARGET_TOKENS` | `800` | Taille cible des passages |
| `CHUNK_MAX_TOKENS` | `1200` | Limite estimée d’un passage |
| `CHUNK_OVERLAP_SEGMENTS` | `2` | Chevauchement des passages SRT |
| `CHUNK_SRT_PAUSE_MS` | `2500` | Pause favorisant une coupure SRT |
| `EXTRACTION_MAX_RETRIES` | `1` | Nouvelle tentative maximale après sortie JSON/sémantique invalide |
| `EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE` | `2` | Maximum de connaissances utiles demandé par passage |
| `QDRANT_PATH` | `qdrant` | Répertoire de l’index dérivé, relatif à `SECOND_BRAIN_DATA_DIR` |
| `EMBEDDING_BATCH_SIZE` | `8` | Nombre de connaissances par appel Ollama `/api/embed` |
| `SEMANTIC_SEARCH_TOP_K` | `5` | Nombre de résultats sémantiques retournés par défaut |
| `RAG_RETRIEVAL_TOP_K` | `8` | Résultats classés récupérés pour une question RAG |
| `RAG_CONTEXT_MAX_NODES` | `5` | Nombre maximal de KnowledgeNodes transmis au générateur |
| `RAG_CONTEXT_MAX_CHARS` | `10000` | Budget total du contexte de connaissances |
| `RAG_KNOWLEDGE_MAX_CHARS` | `1200` | Contenu maximal transmis par KnowledgeNode |
| `RAG_MAX_EVIDENCE_PER_NODE` | `1` | Preuve maximale transmise par KnowledgeNode |
| `RAG_EVIDENCE_MAX_CHARS` | `400` | Taille maximale de chaque extrait de preuve |
| `GRAPH_NEIGHBORS_K` | `8` | Nombre maximal initial de voisins recherchés par KnowledgeNode |
| `GRAPH_MIN_SIMILARITY` | `0.45` | Similarité cosinus minimale avant le petit bonus de tags |
| `GRAPH_TAG_WEIGHT` | `0.05` | Poids maximal du Jaccard des tags dans une relation |
| `GRAPH_PCA_DIMENSIONS` | `50` | Dimension intermédiaire maximale du clustering |
| `CLUSTER_MIN_SIZE` | `5` | Taille cible minimale d’un cluster cohérent |
| `CLUSTER_MAX_DOMAINS` | `6` | Nombre maximal de grands domaines testés |
| `CLUSTER_MAX_THEMES_PER_DOMAIN` | `6` | Nombre maximal de thèmes testés dans un domaine |
| `CLUSTER_MIN_SILHOUETTE` | `0.1` | Qualité minimale requise pour accepter une partition |
| `CLUSTER_NOISE_IQR_FACTOR` | `1.5` | Facteur robuste utilisé pour détecter les nœuds isolés |
| `CLUSTER_LABEL_BATCH_SIZE` | `12` | Clusters nommés par appel Ollama au maximum |
| `CLUSTER_REPRESENTATIVE_COUNT` | `5` | Connaissances représentatives retenues par cluster |
| `UMAP_NEIGHBORS` | `15` | Voisinage utilisé uniquement pour la projection 2D |
| `UMAP_MIN_DIST` | `0.15` | Compacité locale de la projection UMAP |
| `UMAP_RANDOM_STATE` | `42` | Graine de reproductibilité du layout |
| `JOB_STALE_HEARTBEAT_SECONDS` | `120` | Délai sans heartbeat avant de signaler un job `running` comme stale |

Les limites de chunks sont des estimations indépendantes du modèle. Une entrée SRT reste entière tant qu’elle tient sous le plafond ; seule une entrée exceptionnellement longue utilise un découpage de secours borné. Les textes suivent d’abord les paragraphes et les phrases.

`QDRANT_PATH` doit rester à l’intérieur de `SECOND_BRAIN_DATA_DIR`. Ne placez pas une base SQLite ou un index Qdrant actif dans un dossier synchronisé par un service cloud. Toute modification d’un paramètre `GRAPH_*`, `CLUSTER_*` ou `UMAP_*` change le fingerprint de construction et demande un nouveau profil du cerveau ; elle ne modifie pas le profil d’embedding.

## API jusqu’à la Phase 6B

- `GET /api/v1/system/health` : connexion SQLite ;
- `GET /api/v1/system/readiness` : état Ollama et disponibilité du modèle de génération ;
- `POST /api/v1/sources/manual` : note libre ;
- `POST /api/v1/sources/upload` : import TXT/SRT ;
- `GET /api/v1/sources` et `GET /api/v1/sources/{id}` : liste et détail ;
- `GET /api/v1/sources/{id}/segments` : segments SRT ;
- `POST /api/v1/sources/{id}/analyze` : mise en file de l’analyse ;
- `GET /api/v1/sources/{id}/analysis` : dernier job et progression persistante de la source ;
- `GET /api/v1/jobs/{id}` : état détaillé du traitement ;
- `GET /api/v1/sources/{id}/nodes` : connaissances d’une source ;
- `GET /api/v1/nodes/{id}` : connaissance, source et preuves exactes ;
- `GET /api/v1/vector-index/status` : disponibilité du modèle d’embedding, profil, compteurs et job actif ;
- `POST /api/v1/vector-index/index` : indexation incrémentale idempotente ;
- `POST /api/v1/vector-index/rebuild` : reconstruction confirmée dans une nouvelle génération ;
- `GET /api/v1/vector-index/jobs/{id}` : progression persistante d’une indexation ;
- `POST /api/v1/search/semantic` : embedding de la requête, similarité cosinus et résultats enrichis depuis SQLite ;
- `POST /api/v1/rag/answer` : question indépendante, réponse structurée, citations validées, résultats récupérés/utilisés et métriques ;
- `GET /api/v1/brain/status` : profil du cerveau actif, fraîcheur, statistiques et job courant ;
- `POST /api/v1/brain/rebuild` : construction confirmée d’une nouvelle génération dérivée ;
- `POST /api/v1/brain/relabel` : relabeling facultatif du profil actif sans recalcul mathématique ;
- `GET /api/v1/brain/jobs/{id}` : progression persistante d’une construction ou d’un relabeling ;
- `GET /api/v1/brain/clusters` et `GET /api/v1/brain/clusters/{id}` : hiérarchie, enfants, membres et positions ;
- `GET /api/v1/brain/graph` : nœuds, clusters et arêtes bornés au niveau ou cluster demandé, sans vecteurs bruts ;
- `GET /api/v1/brain/search` : recherche textuelle locale des labels de clusters et titres des KnowledgeNodes, avec chemin hiérarchique et coordonnées.

## Structure principale

```text
backend/alembic/                    migrations SQLite
backend/src/second_brain/llm/      client Ollama, schémas JSON et prompts
backend/src/second_brain/pipeline/ chunking indépendant du modèle
backend/src/second_brain/jobs/     worker local persistant
backend/src/second_brain/vector/   embeddings typés et adaptateur Qdrant local
backend/src/second_brain/rag/      contexte, schémas RAG et validation des citations
backend/src/second_brain/graph/    kNN, clustering hiérarchique, labels et projection UMAP
backend/src/second_brain/services/ analyses, index vectoriel et orchestration du cerveau
frontend/src/features/brain/       contrats, Graphology, Sigma, zoom, navigation et panneaux
frontend/                           React, Sources, cerveau, recherche et Paramètres
scripts/                            installation, démarrage et vérifications Windows
.env.example                        configuration documentée
pyproject.toml                      dépendances et outils Python
```

La génération et les embeddings restent entièrement locaux et sont déclenchés par des actions explicites. La Phase 6B ne fait que visualiser le modèle mathématique dérivé de la Phase 6A avec Sigma.js et Graphology. Elle n’ajoute ni nouveau calcul de clustering ou de layout, historique conversationnel persistant, recherche web, API cloud, PDF ou EPUB. Aucune Phase 7 n’est commencée.

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'
$frontendPath = Join-Path $repoRoot 'frontend'
$envExamplePath = Join-Path $repoRoot '.env.example'
$envPath = Join-Path $repoRoot '.env'

if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "La creation de l'environnement Python a echoue. Verifiez que python --version indique Python 3.10.x."
    }
}

$pythonVersion = (& $pythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if (-not $pythonVersion.StartsWith('3.10.')) {
    throw "La Phase 1 attend Python 3.10, mais le venv utilise Python $pythonVersion. Supprimez .venv puis relancez ce script avec Python 3.10 installe."
}

$nodeVersionText = (& node.exe --version).TrimStart('v')
$nodeVersion = [version]$nodeVersionText
$nodeIsSupported = `
    (($nodeVersion.Major -eq 20) -and ($nodeVersion.Minor -ge 19)) -or `
    (($nodeVersion.Major -ge 22) -and ($nodeVersion -ge [version]'22.12.0'))
if (-not $nodeIsSupported) {
    throw "Vite 7 attend Node.js 20.19+ ou 22.12+. Version detectee : $nodeVersionText."
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'La mise a jour de pip a echoue.'
}

Push-Location $repoRoot
try {
    & $pythonPath -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) {
        throw "L'installation des dependances Python a echoue."
    }

    & $pythonPath -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        throw "La migration de la base SQLite a echoue."
    }
}
finally {
    Pop-Location
}

Push-Location $frontendPath
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) {
        throw "L'installation des dependances frontend a echoue."
    }
}
finally {
    Pop-Location
}

Write-Host 'Installation terminee.' -ForegroundColor Green

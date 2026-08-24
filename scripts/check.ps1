[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$frontendPath = Join-Path $repoRoot 'frontend'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Environnement Python absent. Executez d'abord .\scripts\setup.ps1."
}

Push-Location $repoRoot
try {
    & $pythonPath -m ruff check backend
    if ($LASTEXITCODE -ne 0) { throw 'Ruff a detecte des erreurs.' }

    & $pythonPath -m ruff format --check backend
    if ($LASTEXITCODE -ne 0) { throw "Le formatage Python n'est pas a jour." }

    & $pythonPath -m pytest
    if ($LASTEXITCODE -ne 0) { throw 'Les tests backend ont echoue.' }
}
finally {
    Pop-Location
}

Push-Location $frontendPath
try {
    & npm.cmd run test
    if ($LASTEXITCODE -ne 0) { throw 'Les tests frontend ont echoue.' }

    & npm.cmd run lint
    if ($LASTEXITCODE -ne 0) { throw 'Le lint frontend a echoue.' }

    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Le build frontend a echoue.' }
}
finally {
    Pop-Location
}

Write-Host 'Toutes les verifications sont passees.' -ForegroundColor Green

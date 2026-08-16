[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$frontendPath = Join-Path $repoRoot 'frontend'
$logPath = Join-Path $repoRoot 'data\logs'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Environnement Python absent. Executez d'abord .\scripts\setup.ps1."
}

Push-Location $repoRoot
try {
    & $pythonPath -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        throw 'La migration de la base SQLite a echoue.'
    }
}
finally {
    Pop-Location
}

New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$backendOutputLog = Join-Path $logPath 'backend-dev.stdout.log'
$backendErrorLog = Join-Path $logPath 'backend-dev.stderr.log'

$backendArguments = @(
    '-m', 'uvicorn',
    'second_brain.main:app',
    '--host', '127.0.0.1',
    '--port', '8001'
)

$backendProcess = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $backendArguments `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -RedirectStandardOutput $backendOutputLog `
    -RedirectStandardError $backendErrorLog `
    -WindowStyle Hidden

$backendReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($backendProcess.HasExited) {
        break
    }

    try {
        $health = Invoke-RestMethod `
            -Uri 'http://127.0.0.1:8001/api/v1/system/health' `
            -TimeoutSec 1
        if ($health.status -eq 'ok') {
            $backendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 250
    }
}

if (-not $backendReady) {
    if (Get-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    }
    throw "Le backend n'est pas devenu disponible. Consultez $backendErrorLog."
}

Write-Host "Backend pret (PID $($backendProcess.Id)) sur http://127.0.0.1:8001"
Write-Host "Logs backend : $backendOutputLog et $backendErrorLog"
Write-Host 'Le frontend va demarrer sur http://127.0.0.1:5173'

try {
    Push-Location $frontendPath
    try {
        & npm.cmd run dev -- --host 127.0.0.1
        if ($LASTEXITCODE -ne 0) {
            throw "Le serveur Vite s'est arrete avec une erreur."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if (Get-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    }
}

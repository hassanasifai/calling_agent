[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$GoRoot = Join-Path $ProjectRoot 'transport\diago-sidecar'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Run scripts\setup.ps1 first.'
}

Push-Location $GoRoot
try {
    go test ./...
    go build ./...
}
finally {
    Pop-Location
}

Push-Location $ProjectRoot
try {
    & $VenvPython -m black --check nrsp_agent tests scripts
    & $VenvPython -m ruff check nrsp_agent tests scripts
    & $VenvPython -m pytest
}
finally {
    Pop-Location
}

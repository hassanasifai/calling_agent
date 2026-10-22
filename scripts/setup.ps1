[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$EditableSpec = '{0}[dev,voice-engine]' -f $ProjectRoot
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TEMP = Join-Path $ProjectRoot '.tmp'
$env:TMP = $env:TEMP

New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TEMP | Out-Null
if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot '.venv')
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e $EditableSpec

<#
.SYNOPSIS
  One-click Urdu AI calling from this laptop; all inference runs on the VM API.

.DESCRIPTION
  1. Checks local config (config\local\*, Git-ignored; created from templates).
  2. Checks VPN reachability and the VM inference API (/health, with token).
  3. Starts the Go/Diago bridge with live calls DISABLED and checks SIP
     registration.
  4. Lets you pick an agent profile and shows its fictional test data.
  5. Places ONE call only after you type the exact authorization phrase; the
     bridge is restarted allowlisted for that single number and stopped after.
  6. Saves the call result JSON, pipeline log and per-turn latency report in
     .tmp\live\<timestamp>\.

  Nothing is dialled without step 5. Use -CheckOnly to stop after step 3.
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$Local = Join-Path $Root 'config\local'
$Bridge = Join-Path $Root 'transport\diago-sidecar\bin\nrsp-diago-bridge.exe'
$BridgeLog = Join-Path $Root '.tmp\bridge.log'

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "FAILED: $text" -ForegroundColor Red; exit 1 }

function Read-EnvFile($path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
        $line = $line.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $key, $value = $line.Split('=', 2)
            $values[$key.Trim()] = $value.Trim()
        }
    }
    return $values
}

function Read-Setting($toml, $name) {
    $match = Select-String -LiteralPath $toml -Pattern ('^\s*' + $name + '\s*=\s*"?([^"#]+)"?') |
        Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value.Trim() }
    return $null
}

# ---------------------------------------------------------------- 1. config
Step 'Local configuration'
if (-not (Test-Path -LiteralPath $Py)) { Fail 'Python env missing: run scripts\setup.ps1 once.' }
New-Item -ItemType Directory -Force -Path $Local, (Join-Path $Root '.tmp') | Out-Null
$created = @()
$templates = @{
    'windows.toml' = 'config\windows-remote.example.toml'
    'bridge.env'   = 'config\.env.example'
}
foreach ($name in $templates.Keys) {
    $target = Join-Path $Local $name
    if (-not (Test-Path -LiteralPath $target)) {
        Copy-Item -LiteralPath (Join-Path $Root $templates[$name]) -Destination $target
        $created += $target
    }
}
$tokenFile = Join-Path $Local 'windows.env'
if (-not (Test-Path -LiteralPath $tokenFile)) {
    Set-Content -LiteralPath $tokenFile -Encoding ASCII -Value 'NRSP_INFERENCE_TOKEN=paste-the-token-from-the-VM-config/local/vm-api.env'
    $created += $tokenFile
}
if ($created.Count -gt 0) {
    Write-Host 'Created from templates (fill them in, then run again):'
    $created | ForEach-Object { Write-Host "  $_" }
    exit 1
}
$settings = Join-Path $Local 'windows.toml'
$apiUrl = Read-Setting $settings 'remote_api_url'
$inference = (Read-EnvFile $tokenFile)['NRSP_INFERENCE_TOKEN']
if (-not $apiUrl) { Fail "remote_api_url missing in $settings" }
if (-not $inference -or $inference.Length -lt 24 -or $inference.StartsWith('paste')) {
    Fail "set NRSP_INFERENCE_TOKEN in $tokenFile"
}
$bridgeEnv = Read-EnvFile (Join-Path $Local 'bridge.env')
foreach ($key in 'NRSP_SIP_PBX_HOST', 'NRSP_SIP_USERNAME', 'NRSP_SIP_PASSWORD', 'NRSP_SIP_BIND_HOST') {
    if (-not $bridgeEnv[$key] -or $bridgeEnv[$key] -like 'replace*' -or $bridgeEnv[$key] -eq '192.0.2.10') {
        Fail "set $key in config\local\bridge.env"
    }
}
Write-Host "API $apiUrl  PBX $($bridgeEnv['NRSP_SIP_PBX_HOST'])  SIP user $($bridgeEnv['NRSP_SIP_USERNAME'])"

# ---------------------------------------------------------------- 2. VPN + API
Step 'VPN and VM inference API'
$uri = [Uri]$apiUrl
$tcp = Test-NetConnection -ComputerName $uri.Host -Port $uri.Port -WarningAction SilentlyContinue
if (-not $tcp.TcpTestSucceeded) { Fail "cannot reach $($uri.Host):$($uri.Port) - is the VPN connected?" }
$headers = @{ Authorization = "Bearer $inference" }
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $health = Invoke-RestMethod -Uri "$apiUrl/health" -Headers $headers -TimeoutSec 10
} catch {
    Fail "API health check failed: $($_.Exception.Message) (token or allowlist?)"
}
$sw.Stop()
Write-Host ("API status={0} llm={1} voices={2} round-trip={3} ms" -f `
    $health.status, $health.llm, ($health.tts_voices -join ','), $sw.ElapsedMilliseconds)
if ($health.status -ne 'ok') { Fail 'VM API degraded (LLM down?) - on the VM: scripts/vm/ctl.sh status' }

# ---------------------------------------------------------------- 3. bridge + SIP
function Start-Bridge([string]$allowlist) {
    $controlToken = -join ((48..57) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
    foreach ($key in $bridgeEnv.Keys) { [Environment]::SetEnvironmentVariable($key, $bridgeEnv[$key], 'Process') }
    $env:NRSP_BRIDGE_AUTH_TOKEN = $controlToken
    if (-not $env:NRSP_BRIDGE_CONTROL_ADDRESS) { $env:NRSP_BRIDGE_CONTROL_ADDRESS = '127.0.0.1:8765' }
    if ($allowlist) {
        $env:NRSP_LIVE_CALLS_ENABLED = 'true'; $env:NRSP_LIVE_CALL_ALLOWLIST = $allowlist
    } else {
        $env:NRSP_LIVE_CALLS_ENABLED = 'false'; $env:NRSP_LIVE_CALL_ALLOWLIST = ''
    }
    $proc = Start-Process -FilePath $Bridge -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $BridgeLog -RedirectStandardError "$BridgeLog.err"
    return $proc
}
function Test-Registered {
    for ($i = 0; $i -lt 20; $i++) {
        & $Py -m nrsp_agent.devtools.live_call --check-only --out (Join-Path $Root '.tmp\live\check') 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

Step 'SIP bridge (live calls disabled)'
if (-not (Test-Path -LiteralPath $Bridge)) {
    Push-Location (Join-Path $Root 'transport\diago-sidecar')
    try { go build -o $Bridge . } finally { Pop-Location }
}
Get-Process -Name 'nrsp-diago-bridge' -ErrorAction SilentlyContinue | Stop-Process -Force
$proc = Start-Bridge ''
try {
    if (-not (Test-Registered)) { Fail "SIP registration failed - see $BridgeLog (MicroSIP closed? password? PBX IP?)" }
    Write-Host 'SIP registered with the PBX.'
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
if ($CheckOnly) { Write-Host "`nAll checks passed (no call placed)." -ForegroundColor Green; exit 0 }

# ---------------------------------------------------------------- 4. profile + data
Step 'Agent profile'
$profiles = @(Get-ChildItem -Directory (Join-Path $Root 'config\profiles'))
$extra = Join-Path $Local 'profiles'
if (Test-Path -LiteralPath $extra) { $profiles += @(Get-ChildItem -Directory $extra) }
for ($i = 0; $i -lt $profiles.Count; $i++) { Write-Host ("  [{0}] {1}" -f ($i + 1), $profiles[$i].Name) }
$choice = [int](Read-Host 'Profile number')
if ($choice -lt 1 -or $choice -gt $profiles.Count) { Fail 'no such profile' }
$profileDir = $profiles[$choice - 1].FullName
$profileName = $profiles[$choice - 1].Name
$data = Join-Path $Local ("test-call.{0}.json" -f $profileName)
if (-not (Test-Path -LiteralPath $data)) {
    $example = Join-Path $profileDir 'test_call.example.json'
    if (-not (Test-Path -LiteralPath $example)) { Fail "create $data with this profile's call data" }
    Copy-Item -LiteralPath $example -Destination $data
}
Write-Host "Test data ($data) - must be FICTIONAL, never a real customer:"
Get-Content -LiteralPath $data -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }

# ---------------------------------------------------------------- 5. authorization
Step 'Authorize ONE real call'
$number = (Read-Host 'Exact dial string (as MicroSIP dialled it, e.g. 903XXXXXXXXX)').Trim()
if ($number -notmatch '^[0-9+]{4,20}$') { Fail 'dial string must be digits only' }
$phrase = "CALL $number"
$typed = Read-Host "Type '$phrase' to place exactly one call (anything else cancels)"
if ($typed -cne $phrase) { Write-Host 'Cancelled - no call placed.'; exit 0 }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $Root ".tmp\live\$stamp"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:NRSP_INFERENCE_TOKEN = $inference
$proc = Start-Bridge $number
try {
    if (-not (Test-Registered)) { Fail 'SIP registration failed on restart' }
    Step "Calling $number with profile $profileName"
    & $Py -m nrsp_agent.devtools.live_call --number $number --profile-dir $profileDir `
        --call-data $data --settings $settings --out $out
    $code = $LASTEXITCODE
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item Env:NRSP_LIVE_CALLS_ENABLED, Env:NRSP_LIVE_CALL_ALLOWLIST, Env:NRSP_INFERENCE_TOKEN -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath $BridgeLog, "$BridgeLog.err" -Destination $out -ErrorAction SilentlyContinue
}
& $Py (Join-Path $Root 'scripts\turn_latency.py') (Join-Path $out 'pipeline.log') |
    Tee-Object -FilePath (Join-Path $out 'latency.txt')
Write-Host "`nSaved: $out (result JSON, pipeline.log, bridge.log, latency.txt)" -ForegroundColor Green
exit $code

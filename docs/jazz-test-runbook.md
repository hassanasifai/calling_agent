# First real Jazz call: preflight and runbook

Status: prepared, **not executed**. A real call requires the project owner's
explicit authorization for one exact test number. Nothing below changes the
miniSIPServer, MicroSIP or ZTE configuration.

## Path under test

```text
Windows laptop: Python/Pipecat ⇄ Go bridge (SIP UA, ext 100)
      ⇄ miniSIPServer V70 (192.168.100.5:5060 UDP, dial prefix 9)
      ⇄ ZTE Blade V50 "android-gw" (android-voice-sip-agent, Direct telephony)
      ⇄ Jazz SIM ⇄ authorized test handset
```

## Decisions to confirm before the call

1. **SIP identity.** The bridge registers as a normal extension. To avoid any
   PBX change, reuse extension `100` (the one MicroSIP used) and **close
   MicroSIP first**; two UAs registered as `100` would split incoming
   requests. A dedicated extension (e.g. `101`) is cleaner but means editing
   miniSIPServer, which needs separate approval.
2. **Dialed number.** The bridge sends the allowlisted string verbatim as the
   SIP request user, so it must be exactly what MicroSIP dialed successfully:
   `9` + the Pakistani mobile number in the format that worked (for example
   `903XXXXXXXXX`). The allowlist and the dial string must match exactly.
3. **Concurrency.** One ZTE/Jazz SIM carries one cellular call. The bridge
   enforces one active call per bridge instance (`gateway_channel_busy`), and
   one bridge instance maps to one gateway ID (`sim-1`). Do not run a second
   bridge against the same gateway.

## Preflight (no call is placed)

```powershell
Set-Location D:\calling
# 1. Code and offline media verification
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
# 2. Offline end-to-end conversation over real SIP/RTP (loopback PBX)
$env:NRSP_E2E = '1'; .\.venv\Scripts\python.exe -m pytest tests\e2e -q --no-cov
Remove-Item Env:NRSP_E2E
# 3. Free RAM: the stack needs ~3.5 GB resident (measured paging on 7.7 GB
#    made LLM first token 33 s). Close browsers; `wsl --shutdown` if unused.
# 4. Models warm and served from the project store
$env:OLLAMA_MODELS = 'D:\calling\data\models\ollama'; $env:OLLAMA_KEEP_ALIVE = '-1'
Start-Process D:\Ollama\ollama.exe -ArgumentList serve -WindowStyle Hidden
# 5. PBX reachable from this laptop (UDP 5060); confirm the current IP
Test-NetConnection 192.168.100.5 -Port 5060   # TCP probe only; UDP is SIP's transport
```

Also confirm on the ZTE that the agent shows *registered* and audio mode is
*Direct telephony*, and that MicroSIP is closed.

## Bridge start (live switch still off)

Secrets stay in the shell session only; never commit them.

```powershell
Set-Location D:\calling\transport\diago-sidecar
go build -o bin\nrsp-diago-bridge.exe .
$env:NRSP_BRIDGE_AUTH_TOKEN = -join ((48..57 + 97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
$env:NRSP_SIP_PBX_HOST = '192.168.100.5'
$env:NRSP_SIP_PBX_PORT = '5060'
$env:NRSP_SIP_USERNAME = '100'
$env:NRSP_SIP_PASSWORD = '<extension-100-secret>'
$env:NRSP_SIP_BIND_HOST = '<this-laptop-LAN-IP>'   # so SDP advertises a reachable RTP address
$env:NRSP_LIVE_CALLS_ENABLED = 'false'
.\bin\nrsp-diago-bridge.exe
```

Expected log: `SIP account registered`. With live calls disabled a
`start_call` is rejected with `live_calls_disabled` — use this to check the
control path without dialing.

## The authorized call (only after written approval of the number)

Restart the bridge with:

```powershell
$env:NRSP_LIVE_CALLS_ENABLED = 'true'
$env:NRSP_LIVE_CALL_ALLOWLIST = '9<authorized-number>'
```

Then run one session with a test customer record whose CNIC digits the
authorized tester knows (no real borrower data). A thin runner for this is
the next implementation step; it calls `run_outbound_session` exactly as the
loopback harness does, with `DiagoTelephonyPort` pointed at the bridge.

## What to capture

- Bridge log (registration, DIALING/RINGING/ANSWERED/RTP_READY timing,
  `media clock re-aligned` warnings, hangup reason).
- Session result JSON: transcript, outcome, per-service TTFB metrics.
- miniSIPServer call log and a Wireshark capture of SIP + RTP on the laptop
  interface (codec negotiated, ptime, RTP from the ZTE flowing both ways).

## Debugging map

| Symptom | First check |
|---|---|
| `sip_not_registered` | PBX IP/port, ext 100 secret, MicroSIP closed, firewall for UDP 5060 |
| `sip_403`/`sip_404` on dial | dial string vs. miniSIPServer prefix-9 route |
| Answered, no audio either way | `NRSP_SIP_BIND_HOST` must be the LAN IP, not 0.0.0.0; Windows firewall for the RTP port range |
| Audio one way only | ZTE "Direct telephony" mode; RTP from ZTE reaching laptop (Wireshark) |
| `rtp_timeout` hangup | ZTE stops RTP during silence (DTX)? raise `NRSP_RTP_TIMEOUT_SECONDS` |
| Bot answers previous turn | ASR slower than `user_turn_stop_timeout_seconds`; check memory/CPU |
| Choppy bot audio | `media clock re-aligned` in bridge log → host starved (RAM/CPU) |

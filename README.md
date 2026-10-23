# NRSP Windows Urdu Voice Agent

Production-oriented Windows-native voice agent code for the proven
miniSIPServer → ZTE Voice SIP Agent → Jazz SIM path. Go/Diago owns SIP, RTP,
G.711 and media timing. Python/Pipecat owns conversation orchestration and
local AI services.

No live call is enabled by default. The bridge rejects dialing unless both
`NRSP_LIVE_CALLS_ENABLED=true` and an exact number allowlist are present.

## Windows setup

```powershell
Set-Location D:\calling
python -m venv .venv
$env:PIP_CACHE_DIR = 'D:\calling\.cache\pip'
$env:TEMP = 'D:\calling\.tmp'
$env:TMP = 'D:\calling\.tmp'
.\.venv\Scripts\python.exe -m pip install -e '.[dev,voice-engine]'
```

Build and verify without a telephone call:

```powershell
Set-Location D:\calling\transport\diago-sidecar
gofmt -w (Get-ChildItem -Filter *.go | ForEach-Object FullName)
go test ./...
go build -o bin\nrsp-diago-bridge.exe .

Set-Location D:\calling
.\.venv\Scripts\python.exe -m black --check nrsp_agent tests scripts
.\.venv\Scripts\python.exe -m ruff check nrsp_agent tests scripts
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\benchmark_local_speech.py
```

Offline end-to-end conversation (real Go bridge, real SIP/RTP to a loopback
test PBX/callee, real Whisper/Qwen3/Piper; no telephone network reachable):

```powershell
# Serve the project's model store; a user-level OLLAMA_MODELS elsewhere would
# start Ollama with no models.
$env:OLLAMA_MODELS = 'D:\calling\data\models\ollama'; $env:OLLAMA_KEEP_ALIVE = '-1'
Start-Process D:\Ollama\ollama.exe -ArgumentList serve -WindowStyle Hidden

.\.venv\Scripts\python.exe -m nrsp_agent.devtools.loopback happy_path   # one scenario, digest
.\.venv\Scripts\python.exe -m nrsp_agent.devtools.loopback all          # every scenario
$env:NRSP_E2E = '1'; .\.venv\Scripts\python.exe -m pytest tests\e2e -q --no-cov
```

Per-scenario reports (transcript, outcome, RTP audit, latency, recordings)
land in `.tmp\e2e\<scenario>\`. Caller audio fixtures are synthetic
(`scripts\make_caller_fixtures.py`) and are not an accuracy benchmark. The
stack needs ~3.5 GB of free RAM; on the 7.7 GB laptop, paging was measured to
make the LLM's first token 33 s. See `docs\jazz-test-runbook.md` before any
real call.

The model benchmark downloads only local model weights into the ignored
`data/models` directory. `config/demo.toml` is the laptop profile;
`config/server.toml` is the starting Windows Server profile and must be sized
with target-server benchmarks.

## Agent profiles (task configuration)

A call's behaviour comes from a profile directory, not from code:
`config/profiles/nrsp_verification/` holds `profile.toml` (persona, objective,
rules, actions, approved texts, examples, voice, ASR prompts) and knowledge
files. Call data is passed per call; only `public_fields` are shown to the
LLM. Build an engine with `nrsp_agent.agent.build_engine(profile_dir, data)`
and pass it to `run_outbound_session(..., engine=...)`. See
`tests/fixtures/profiles/clinic_reminder/` for a second, unrelated task.

## Ubuntu development (CPU, no services)

```bash
uv venv --python 3.13 .venv && uv pip install -e '.[dev,voice-engine]'
scripts/convert_asr_model.sh          # whisper-small-urdu -> CTranslate2 int8
scripts/run_llm_server.sh &           # llama.cpp, loopback :18431, Gemma-4-E2B
python -m nrsp_agent.devtools.loopback all --profile config/ubuntu-dev.toml
NRSP_E2E=1 NRSP_VOICE_CONFIG=config/ubuntu-dev.toml pytest tests/e2e -q --no-cov
```

Benchmarks (run sequentially; `scripts/pressure_check.sh` gates heavy work):
`build_asr_eval.py`, `benchmark_asr.py`, `benchmark_llm_turns.py`,
`simulate_dialogues.py`, `benchmark_e2e_understanding.py`, `benchmark_tts.py`.
Results and decisions: `PROGRESS.md`, `DECISIONS.md` (ADR-007..010).

## Bridge configuration

Copy `config/.env.example` to an untracked `.env`, then set secrets locally.
Do not set the live-call switch during offline development.

```powershell
$env:NRSP_BRIDGE_AUTH_TOKEN = '<at-least-16-random-characters>'
$env:NRSP_SIP_PBX_HOST = '<miniSIPServer-IP>'
$env:NRSP_SIP_USERNAME = '<extension>'
$env:NRSP_SIP_PASSWORD = '<secret>'
$env:NRSP_LIVE_CALLS_ENABLED = 'false'
Set-Location D:\calling\transport\diago-sidecar
.\bin\nrsp-diago-bridge.exe
```

A real test additionally requires explicit authorization for the exact number;
only then set `NRSP_LIVE_CALLS_ENABLED=true` and
`NRSP_LIVE_CALL_ALLOWLIST=<exact-number>`.

See `STATUS.md`, `DECISIONS.md`, and `docs/architecture.md` for verified scope
and production boundaries.

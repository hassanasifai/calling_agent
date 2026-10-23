# PROGRESS

Updated: 2026-10-23 — Ubuntu AIUAT dev VM (CPU only; cgroup 6 CPU / 32 GB;
host shared with busy Financial AI llama-servers, load 4–27 during tests).

## Overall objective
A reusable, configurable AI voice-calling engine: Go/Diago SIP/RTP + Pipecat +
Silero VAD + local Urdu ASR + local LLM + Piper TTS. A task is an **agent
profile** (persona, objective, rules, knowledge, actions, approved texts,
result fields, voice); the core pipeline does not change per task. NRSP
installment reminder with identity verification is the first profile.

## Completed and verified (tested on Ubuntu)
- **Generic engine** `nrsp_agent/agent/`: profile schema/loader, BM25
  knowledge, gated actions (say / record / verify / end), structured result,
  confirmations for irreversible endings, safety actions, keypad entry.
  NRSP lives only in `config/profiles/nrsp_verification/`; a second, unrelated
  profile (`tests/fixtures/profiles/clinic_reminder/`) runs unchanged.
- **LLM-first understanding**: one streamed call per turn, JSON-schema
  constrained to the actions allowed *now*; approved-text actions forced to an
  empty `say` (stops generation early); action resolved as soon as named.
  Code enforces consequential steps; private call data never reaches the model
  (approved texts are rendered by code; history keeps the model's own output).
- **Verification**: digits parsed by code from the transcript (Urdu, English,
  Roman Urdu number words), read back for confirmation; never reconstructed
  by the LLM. **DTMF keypad entry** (RFC 2833) end-to-end: bridge forwards
  keys → engine compares exactly; key values never logged.
- **Prefix caching**: system prompt identical across calls of a profile
  (per-call details are the last lines; 4,884/4,892 chars shared), full in-call
  history (no sliding window): 0 cache misses in dialogue tests.
- **Stale-answer protection**: one LLM request in flight; superseded requests
  are interrupted and late answers dropped by request id.
- Official llama.cpp b11516 CPU build (`.cache/llama.cpp`, AVX-512 kernels):
  4.3× faster generation than the shared OpenBLAS binary.
- Tests: Python 78 passed / 9 skipped (E2E opt-in, Windows-only); Go `go test
  ./...` passes (bridge + media integration incl. new DTMF reader wiring).

## Reliability + latency pass (2026-10-23, later) — done
Sequential: reproduce -> diagnose -> minimal fix -> test -> measure.

### Diagnosed (offline replay of recorded call audio, `scripts/replay_vad.py`)
1. **opt_out miss = input framing bug (proven, fixed).** Pipecat's
   `SOXRStreamAudioResampler` (HQ) emits nothing for four 20 ms frames, then a
   1,660-sample burst. Pipecat's `VADAnalyzer._run_analyzer` scores every
   32 ms window in a call but promotes STARTING->SPEAKING only after the
   whole burst, so a weak last window erased a valid 3-window start. Replaying
   the failed run's caller audio (`.tmp/e2e/opt_out/sent.wav`): the "جی ہاں"
   at 25.7 s gave **no** speech start before, a 25.96-26.48 s segment after.
   Fix: `DiagoInputTransport` re-frames resampled audio to 20 ms (audio is
   byte-identical; VAD now sees <= 1 window per frame).
2. **Residual intermittent misses = Silero reset at speech onset.** Pipecat
   resets Silero's recurrent state every ~2 s of quiet audio; when the reset
   coincides with the start of an answer the model's confidence stays ~0.5
   for the whole answer (a 2 s "نہیں، یہ غلط نمبر ہے" was missed). Phase
   dependent (±50 ms shift detects it) -> explains "intermittent". Stress
   replay, 480 answers after a previous turn + 3-6 s gap, 10 phases, 80 noise
   trials: conf/start 0.5/0.1 s missed 22 (0 false starts); 0.5/0.064 17 (0);
   0.45/0.064 12 (1); **0.4/0.064 8 (1, babble clip)**; 0.35/0.064 8 (2).
   Disabling the reset was worse (11 vs 6 misses on a 4 s-gap set); dither
   and more frequent resets did not help. Default changed to 0.4 / 0.064 s.
   Not fully solved: ~1.7% of answers in this stress set are still missed;
   the engine's silence prompt ("کیا آپ میری آواز سن رہے ہیں؟") is the recovery.
3. **Approved texts waited for the end of the LLM stream (latency, fixed).**
   Pipecat's sentence aggregator holds a sentence ending in "؟"/"۔" until it
   sees a following character; after an approved text nothing follows, so TTS
   started only at LLM end. In the final run 8 of 16 turns had
   `tts_start == llm_end`, 1.4-3 s after the action was known. Fix: the guard
   sends approved texts as `TTSSpeakFrame` (spoken immediately); model replies
   still stream as LLM text.
4. **LLM prompt time is mostly a fixed per-request cost** (server log): 1 or
   8 prompt tokens took 2.8-3.1 s, 87 tokens 2.2 s. Under investigation with
   `scripts/probe_llm_cache.py` (cache rewind on Gemma's SWA cache suspected).

Baseline per-turn breakdown of the previous final run (`scripts/turn_latency.py`,
seconds after VAD stop, n=16): ASR 2.69 P50 / 3.36 P95; LLM first token 5.01 /
6.27; TTS start 7.28 / 9.89; first bot audio 8.07 / 10.75.

Tests after fixes 1-3: Python 78 passed / 9 skipped.

5. **LLM prefix prefill (latency, implemented).** On a quiet host prompt cost
   is per token (~26 ms/token, 5 threads; an identical repeat costs 0.18 s —
   the earlier 3 s single-token requests were host contention). The model's
   own last output (~23 tokens) is re-processed every turn: grammar-constrained
   decoding emits non-canonical token splits, so it never matches the
   re-tokenised history (template verified consistent). Everything except the
   caller's words is known when the bot's reply is decided, so
   `ConversationController` now sends that prefix (system + history +
   `CallEngine.turn_prefix()`) to the server while the bot speaks
   (`llm_prefill = true`, configurable). `scripts/probe_llm_cache.py`, 4 turns:
   first token 2.18-2.65 s -> 0.79-1.10 s; prompt tokens on the turn 66-90 ->
   11-23; same actions chosen. `turn_message` output is byte-identical.
6. `nrsp_agent/devtools/live_call.py`: thin runner for ONE session through an
   already running bridge (the runbook's missing step). Not run against a real
   PBX (not authorized).

### SIP/RTP loopback after fixes 1-5 (`.tmp/e2e_fix1`, `.tmp/loopback_fix1.log`)
**9/9 scenarios reached the correct outcome** (before: 7/9): happy_path,
keypad_verification (DTMF), barge_in, opt_out, wrong_person, handoff, silence,
remote_hangup_mid_turn, rtp_loss. Media: 0 dropped caller frames.
Latency, peer clock, caller end of speech -> first bot audio, n=18:
**P50 5.92 s, P95 10.02 s, max 11.12 s** (before: P50 8.1, P95 11.7).
Per stage after VAD stop (n=18): ASR 2.52 P50 / 2.99 P95; LLM first token
3.17 / 3.82 (was 5.01 / 6.27); TTS start 4.40 / 8.48 (was 7.28 / 9.89).
The slow tail is turns where the model writes its own reply ("say"): the reply
is generated at ~8-10 tok/s before the first sentence can be spoken.
barge_in: the peer reported no `bot_stop_ms` because the caller started in a
pause between greeting sentences (2620-3023 ms); the bot resumed and stopped at
3560 ms, i.e. 740 ms after caller onset (600 ms before) — measurement gap in
the peer's annotate(), not a regression.
Remaining understanding miss: barge_in's last caller turn "ٹھیک ہے، شکریہ،
میں لا حافظ" (ASR of "...اللہ حافظ") was recorded as payment difficulty
(previous run: an off-topic model reply) — an LLM decision error.

### LLM decision reliability (new held-out set, nothing adopted)
New set `tests/fixtures/intent/conversational.jsonl` (44 turns: short replies,
corrections, Roman Urdu/code-switching, garbled ASR from the failed runs,
unexpected answers, follow-up questions, closings) with the bot's real pending
question per stage (new benchmark stages readback / closing / confirm_wrong;
a label may list equally acceptable actions). Gemma-4-E2B, 5 threads:
**baseline 36/44 = 81.8 %**. Worst error: during the wrong-person
confirmation "نہیں نہیں میں ہی احمد ہوں" -> affirm (would end a call with the
right person). Others: garbled identity / "ہیلو؟ ہیلو؟" -> wrong_person
(caught by the confirmation), clarifying questions during verification ->
verify_cnic (code finds no digits and re-asks), loan questions ->
tell_installment. Closings are correct (5/5) once "anything else?" was asked;
the old dev/held-out "finish" misses are mostly labelled under the
pay-on-time question, where "نہیں ..." legitimately reads as difficulty.
Tried: generic reject wording ("or contradicts what you asked") + narrower
NRSP tell_installment description -> conversational 84.1 %, but held-out
82.1 -> 76.9 % and dev 75 -> 70 %. **Reverted** (no net improvement).

## Remote inference deployment: VM API + one-click Windows calling (2026-10-23)
Architecture (no SSH tunnel, no Docker, no new framework, no reboot):
- **VM 10.20.30.111** runs all inference permanently as *user* systemd units
  in `calling-ai.slice` (CPUQuota 600 %, MemoryMax 12 G, Nice 5, Restart=always):
  `calling-llm.service` (existing `scripts/run_llm_server.sh`, llama.cpp on
  127.0.0.1:18431, 5 threads) and `calling-api.service`
  (`nrsp_agent/inference_api.py`, aiohttp, bound to **10.20.30.111:18440 only**):
  `GET /health`, `POST /v1/asr` (PCM16 16 kHz -> text; same
  `transcribe_pcm16()` decoding as local mode), `POST /v1/tts` (Piper aegis,
  streamed WAV per sentence), `/v1/chat/completions` + `/v1/models` proxied to
  the loopback llama-server. Every request: Bearer token (constant-time
  compare) + client IP allowlist; body/text/prompt size caps; one ASR and one
  TTS job at a time; upstream timeouts. Financial AI services untouched.
- **Windows laptop** runs Go/Diago, Pipecat orchestration and Silero VAD only
  (`[voice] inference = "remote"`): `RemoteWhisperSTTService` (subclass of the
  existing Whisper service, keeps the engine's per-turn ASR prompt switch),
  Pipecat's own `PiperHttpTTSService`, the existing OpenAI-compatible LLM client
  (token as api_key); prewarm/prefill send the token too. No model weights are
  loaded. `remote_api_url` must be `http://<private IP literal>:<port>`
  (hostnames/public IPs rejected); the token comes from `NRSP_INFERENCE_TOKEN`.
- `start-calling.ps1` (repo root): config check -> VPN TCP check -> API
  `/health` with token -> bridge with live calls **disabled** + SIP
  registration check -> profile menu (`config/profiles/*`, plus
  `config/local/profiles/*`) -> fictional test data (`test_call.example.json`
  in each profile, copied to `config/local/test-call.<profile>.json`) -> exact
  dial string + typed phrase `CALL <number>` -> bridge restarted allowlisted
  for that one number -> `nrsp_agent.devtools.live_call` (warms the profile's
  system prompt, runs one session) -> bridge stopped -> result JSON,
  pipeline.log, bridge logs and `latency.txt` in `.tmp\live\<timestamp>\`.
  `-CheckOnly` stops after the SIP check.

### Verified on the VM (2026-10-23)
- `scripts/vm/setup.sh 25.1.0.114/32` (25.1.0.114 = the only current VPN
  client, the VNC session): units enabled + started; linger already `yes`.
  `config/local/` (token file mode 600) is Git-ignored. For the tests below
  `10.20.30.111/32` was added to `allowed_clients` (remove it if unwanted;
  then `ctl.sh health` reports "client not allowed", which is expected).
- Security tests (`tests/test_inference_api.py`, 17): no/wrong token -> 401,
  client outside allowlist -> 403, public bind / 0.0.0.0/0 allowlist / public
  or hostname client URL refused, path-like voice and oversize text -> 400.
  Live: the VM's own address was refused (403) before it was allowlisted.
- Crash recovery: `kill -9` of both service processes -> both active and
  `/health` ok again after 11 s (NRestarts=1).
- **Boot start: NOT verified by reboot** (forbidden). It relies on the units
  being enabled under `default.target` with linger = yes.
- All 9 SIP/RTP loopback scenarios with the client in **remote mode** (no
  weights loaded, all inference over 10.20.30.111:18440): 8/9 correct;
  keypad_verification ended wrong_person on the known ASR/LLM miss (Whisper:
  "یہ ہند میں احمد علی بات کر رہا"; Gemma -> wrong_person), not remote-specific.
  Latency caller end -> first bot audio, n=17: **P50 5.30 s, P95 6.36 s,
  max 8.60 s**. Client ASR round trip 2.301 s P50 vs server decode 2.299 s;
  client LLM first token 0.63 s P50 (prefill works through the proxy); TTS
  first audio 0.44 s P50 (client) / server 0.438 s. Network overhead on the
  VM itself ~0; **over the real VPN it is UNVERIFIED** (measure from Windows).
- Launcher Python path against the loopback PBX with the **clinic profile**
  (unchanged engine, remote inference): `live_call --check-only` ->
  `sip_registered=true`; one call -> outcome completed (identity_ok,
  attendance_known) despite garbled ASR; reply latency 4.0-5.5 s; 0 RTP gaps.
- `start-calling.ps1` itself was **not executed** (no PowerShell on the VM).

### Exact setup
VM (once; idempotent; no sudo, no reboot):
```bash
cd ~/calling_agent
scripts/vm/setup.sh 25.1.0.114/32     # laptop's VPN address/CIDR
scripts/vm/ctl.sh status              # start|stop|restart|status|logs|health
cat config/local/vm-api.env           # token to copy to the laptop
```
Allowed clients: edit `config/local/vm-api.toml`, then
`systemctl --user restart calling-api.service`.

Windows (once): pull this repo version, `scripts\setup.ps1`, then run
`.\start-calling.ps1` — the first run creates `config\local\windows.toml`
(API URL), `config\local\windows.env` (paste `NRSP_INFERENCE_TOKEN=...`) and
`config\local\bridge.env` (PBX 192.168.100.5, extension 100 + password,
`NRSP_SIP_BIND_HOST` = laptop LAN IP) and stops; fill them in and run again.
`.\start-calling.ps1 -CheckOnly` verifies VPN, API and SIP without calling.

## First real Jazz call: FAILED transport_error (2026-10-23)
- Symptom: DIALING, then FAILED after 10.2 s with `transport_error`, no SIP
  status, no RINGING/ANSWERED/RTP_READY, no bot audio.
- Root cause of the *label*: sipgo `WaitAnswer` returns `*ErrDialogResponse`
  (pointer) but `classifyError` matched only the value type, so every final
  SIP rejection (403/404/480/486/488/503...) was reported as `transport_error`
  with no status. Reproduced on loopback: a gateway answering 100 then 480
  gave exactly `FAILED transport_error` with no sip_status on the old code;
  fixed code gives `sip_480` + `sip_status=480`
  (`sip_failure_test.go`, `TestClassifyErrorReportsSIPStatusForBothErrorForms`).
- 10.2 s with no 180/183 fits a gateway/PBX final rejection after GSM call
  setup (it is far below Timer B, 32 s, and the 90 s invite timeout); the
  actual status is UNKNOWN until the next authorized call.
- Diagnostics added (bridge stderr -> `.tmp\bridge.log.err`, copied into the
  run folder): `INVITE start` (masked number, PBX, bind, timeout, offered
  codecs), one `SIP response` line per response (status, reason, Reason/
  Warning/Server headers incl. Q.850 cause, Via received/rport as seen by the
  PBX, answered `m=audio`/`c=` SDP lines, auth challenge flag), and
  `INVITE failed` (elapsed, stage: no_response / auth_challenge / provisional
  / answered_media_setup / final_response, last status, redacted error).
  Registration failures now log the redacted error text too. No credentials,
  digests or full numbers are logged.
- `start-calling.ps1` now rebuilds the bridge when Go sources are newer than
  the .exe (it previously reused a stale binary after a pull).
- Go vet/build (linux + windows/amd64) and `go test ./...` pass.

## Remaining tasks (priority)
0. On Windows: run `.\start-calling.ps1 -CheckOnly` (UNVERIFIED: PowerShell
   script, VPN latency, SIP registration), then ONE authorized Jazz call.
1. Windows: pick the Ollama tag for Gemma-4-E2B (or run llama-server), confirm
   the ZTE/Jazz gateway forwards RFC 2833 DTMF, re-measure latency there.
2. Real recorded Pakistani telephone speech for an accuracy set (needs data).
3. Diagnose the intermittent live miss of short answers (per-frame VAD logging).
4. Latency: needs faster hardware for ASR + LLM (GPU or dedicated CPU); on CPU,
   consider whisper-base-urdu for non-verification turns only if accuracy loss
   (85% -> 74% understanding) is acceptable.
5. Human listening test of aegis vs fasih samples.

## Issues and blockers
- No real Pakistani *telephone* recordings: FLEURS is read speech; the domain
  set and loopback callers are synthetic (TTS). Field accuracy is unverified.
- Spoken 4-digit CNIC over G.711 is unreliable for every CPU ASR measured
  (≤25% per attempt on non-bot voices) → keypad entry added; read-back retained.
- CPU latency: end of caller speech → first bot audio is ~6.5–9 s on this
  contended host (target <2 s **not met**). Dominant costs: ASR ~2.7 s, LLM
  ~2.5–3 s to the action. A GPU server or dedicated CPU would cut both.
- llama.cpp server on loopback 18431 now runs permanently as the user unit
  `calling-llm.service` (behind the private API); Ollama is not needed.
- Open understanding misses: garbled identity answers -> wrong_person
  (confirmation catches it, but a scripted caller then says "جی ہاں"), and
  ~1.7 % of short answers after a pause missed by VAD in stress replay.

## Benchmarks (all sequential, pressure-gated)
### ASR — telephone band (8 kHz G.711 A-law), 4 threads
| Model | Domain P50/P95 s | Domain WER/CER % | FLEURS (human) WER/CER % | Noise halluc. | RSS MB |
|---|---|---|---|---|---|
| Canopy-M (CTC 58M) | 0.12/0.15 | 40.9/16.3 | 49.3/20.9 | 8/8 (1-char) | 727 |
| Whisper-small (old baseline) | 2.56/3.04 | 32.3/14.5 | 43.8/20.1 | 0/8 | 1006 |
| **whisper-small-urdu b5 (selected)** | 2.66/3.76 | **24.5/10.4** | **28.3/11.5** | 8/8 | 557 |
| whisper-small-urdu b1 | 2.77/3.50 | 25.9/10.6 | 29.7/11.9 | – | – |
| whisper-base-urdu | 0.99/1.15 | 37.2/14.8 | 43.2/18.0 | 7/8 | 275 |
| whisper-tiny-urdu | 0.61/0.73 | 45.9/20.4 | 60.7/27.3 | 3/8 | 242 |

End-to-end understanding (ASR transcript → Gemma-4-E2B action), 68 synthetic
utterances from the two non-bot voices: ground truth 66/68, **whisper-small-urdu
58/68**, whisper-base-urdu 50/68, Canopy-M 46/68.
Digit-grammar CTC decoding on Canopy (experiment, `scripts/ctc_digit_grammar.py`):
held-out digits 0/16 → 4/16; not integrated (insufficient gain).

### LLM — production prompt/schema, 4 threads
| Model | Held-out action acc | Dev acc | False ending | Action P50/P95 s | Reply safety |
|---|---|---|---|---|---|
| **Gemma-4-E2B-it QAT Q4_0 (selected)** | **82.1%** | 75% | 2 (confirm-guarded) | 3.26/5.40 | follows rules |
| Qwen3.5-2B Q4_K_M | 82.1% | 72.5% | 0 | 2.35/3.14 | **unsafe**: "I am a human", "we can give more time", invented facts |
| Qwen3-1.7B Q4_K_M (old default) | 64.1% | 77.5% | 1 | 2.96/3.57 | canned, Hindi danda |
| Qwen2.5-1.5B (early prompt) | – | 47.5% | 16 | 2.1 | poor Urdu |
Multi-turn dialogues (8 unseen, real model): 7/8 pass; 0 prefix-cache misses.
Speed (llama-bench, 4 thr): Gemma-4-E2B pp 43 / tg 8.6 tok/s (6 thr: 51 / 10.4).

### TTS — Piper, G.711 round trip scored by whisper-small-urdu
| Voice | First audio P50/P95 s | RTF | Intelligibility CER |
|---|---|---|---|
| ur_PK-fasih-medium (old) | 0.46/0.82 | 0.18 | 8.7% |
| **ur_PK-aegis_female-medium (selected)** | 0.47/0.86 | 0.19 | **3.8%** |
Rejected for CPU real time: Kani-TTS-Urdu (backbone 42–49 tok/s vs 50 needed,
before codec), Orpheus-Urdu 3B (author: "not for production"). Naturalness
needs human listening: samples in `data/eval/tts_samples/` (git-ignored).

## Architecture decisions (see DECISIONS.md)
Keep Go/Diago + Pipecat + Silero; profile-driven engine; LLM decides, code
validates; constrained JSON; confirmations for irreversible endings; keypad
verification; cross-call-static prompts; no new services.

## Latest test results (2026-10-23, after the reliability pass + remote API)
- Python 96 passed, 9 skipped (coverage 80.0 %); ruff/black clean; Go
  `go test ./...` passes.
- Loopback local mode 9/9; remote mode 8/9 (known understanding miss); clinic
  profile via launcher path: completed. Latency P50 5.3-5.9 s, P95 6.4-10.0 s.

## Earlier test results
- Python 78 passed, 9 skipped; ruff/black clean. Go `go test ./...` passes.
- SIP/RTP loopback, final run: 7/9 scenarios pass (happy_path, barge_in,
  wrong_person, handoff, silence, remote hang-up, RTP loss). keypad_verification
  passed in an earlier run (DTMF over RTP verified) but failed in the final run
  on a garbled-ASR understanding miss; opt_out failed once on the intermittent
  short-answer VAD miss (passed in 2 of 3 runs). Media clean in all calls.
- Latency (18 turns): end of caller speech -> first bot audio P50 8.1 s,
  P95 11.7 s; ASR 2.65 s, LLM first token 2.3 s, TTS 0.54 s (P50). <2 s target
  NOT met on this shared CPU.

## Immediate next action
On the Windows laptop (user action): pull, `scripts\setup.ps1`, fill the three
`config\local` files, `.\start-calling.ps1 -CheckOnly`, then one explicitly
authorized Jazz call; send back `.tmp\live\<timestamp>\`. Real-phone
behaviour (DTMF through the ZTE gateway, echo/barge-in, VPN latency) is
UNVERIFIED until then.

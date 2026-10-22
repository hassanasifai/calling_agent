# NRSP Urdu AI Calling System — Updated Codex Execution Contract
**Edition:** 2026-10-23 | **Workspace:** `D:\calling` | **Mode:** production-oriented development, authorised test calls only

## Your assignment
Act as senior real-time voice/telephony engineer, system integrator, and performance engineer. **Inspect the existing workspace before modifying anything**. Reuse the already demonstrated SIP-to-Jazz telephony and the 13-file Go/Python Diago sidecar scaffold. Prefer adapting an established OSS conversational voice-agent engine (Dograh first; Pipecat directly if Dograh proves impractical on this host) to writing a voice-agent platform from scratch. Deliver an operable, maintainable, scalable system, not a throwaway demo. Keep engineering minimal and measure performance. Execute work, run automated tests, and record what you actually proved. Never describe fake tests as successful real calls.

## Ground truth supplied by the operator (DO NOT lose these accomplishments)
- Windows 11 development laptop, VS Code workspace `D:\calling`; Python 3.13.3. No Ubuntu/remote VM needed, and the Android handset cannot access the corporate Ubuntu VM.
- On Windows, **miniSIPServer V70**, SIP server address `192.168.100.5:5060` (verify this is still the correct LAN address), **MicroSIP** extension `100`, and a ZTE Android phone with **Voice SIP Agent + Shizuku + Jazz SIM** are already working together.
- Real outbound cellular call through MicroSIP → miniSIPServer → ZTE → Jazz succeeded, and real **two-way audio** succeeded after changing Android audio setting to **Direct telephony**, with handset local mute **OFF**.
- MicroSIP CLI dialing from PowerShell worked using the actual running executable path; a manually dialed `9` routing prefix was removed by miniSIPServer before sending the Pakistani `03...` number to the phone.
- MiniSIPServer displayed a misleading concurrency error at external-line **Maximum concurrent outgoing calls = 1**. Setting that PBX parameter to **2** permitted ONE real call; **do not reset the PBX parameter to 1 without isolating the cause**. The physical ZTE/SIM is ONE simultaneous call channel. Enforce one channel via application gateway semaphore, independent of PBX setting.
- Android gateway registers as `android-gw` to miniSIPServer using `pjsua2`, UDP/5060; cellular audio routing uses Direct telephony. Shizuku needs recovery after phone reboot.
- We will retain the existing installation for benchmarking and regression testing; do not uninstall it or change its credentials/configuration blindly. Rotate any previously exposed passwords.
- Codex previously generated **13 files (+1,339 lines)** including `nrsp_agent/telephony/contracts.py`, `nrsp_agent/telephony/diago.py`, `transport/diago-sidecar/{bridge,config,main,protocol}.go`, `go.mod`, Python tests and `pyproject.toml`. **These files are valuable and must be inspected, compiled, tested, and extended rather than deleted.** The scaffold's mock tests DO NOT prove actual SIP/RTP interoperability yet.

## Non-negotiable product constraints
1. **Free/open-source software and local inference only**, no paid APIs, no usage-based AI billing, no paid SIP trunk, no subscriptions. Hardware, electricity, data and the existing Jazz package are available. Distinguish free-to-download from perpetual commercial-use rights, OSS copyleft obligations, model/voice licenses, and free-tier quotas. No reliance on vendor-granted promotional free credits.
2. **Prioritize Windows 11** for active development. Do not require an Ubuntu VM, remote machine, Asterisk, or Docker by assumption. Dograh officially prefers Docker: inspect existing Docker/WSL, actual free disk/RAM, and its local-dev options before deciding. If Dograh cannot realistically run within available Windows resources without an unwanted new OS stack, use its underlying existing Pipecat engine/modules directly or an equivalent proven OSS engine with the **same Go media adapter**. Explain and document the evidence for this fallback; avoid installing giant dependencies simply to check feasibility.
3. **Do not build a new framework** for VAD/turn taking, barge-in, WebSocket transport, workflow UI, ASR/TTS/LLM integration if a maintained open-source library already provides it. Minimal custom code should be limited to the SIP gateway integration and NRSP business rules, plus operational glue.
4. **Efficient real-time media:** Keep SIP, RTP, G.711 encode/decode, jitter/buffering, monotonic timing and pacing in Go via Diago. Python is acceptable for orchestration if measured end-to-end latency permits; local model runtimes usually execute native CPU/GPU kernels. Profile first, optimize measured hot spots; don't replace a functioning AI framework merely because it uses Python.
5. **Scalable by gateway count:** one active call per physical ZTE/SIM, N separate channels can yield N concurrent calls if CPU/GPU and Wi-Fi permit. Multi-call isolation, admission control, per-call VAD/ASR/TTS sessions, queue idempotency and gateway health are required in the design from day one; implement initially with one channel.
6. This is a financial customer-verification/loan-contact workflow. Approved test numbers only during engineering; no uncontrolled dialing. Ensure AI introduction, identity verification before discussing private loan information, opt-out handling, calling windows, data minimization, human escalation, and auditable outcomes. Never fabricate outstanding balances, payment status, or promises. Disable recording by default. Treat sensitive customer data as confidential.
7. Prefer existing Windows installation of MicroSIP/miniSIPServer only for lab regression while evaluating production software licensing. Do **not** assume the miniSIPServer free edition can be deployed commercially. Plan a SIP-compatible, appropriately licensed replacement or obtain a commercial license **before** deployment; never swap the PBX prematurely and lose working calls.
8. Real outbound calls require a feature flag, an exact allowlist of operator-owned or explicitly approved test numbers, a maximum-call cap, and an explicit human-initiated command. Mock tests must never accidentally dial. Mask numbers and credentials in logs.

## Repositories and research to validate against real source code
- Dograh: https://github.com/dograh-hq/dograh — open-source BSD-2-Clause, workflow/campaign UI, Pipecat runtime, published custom telephony-provider mechanism. Review `api/services/telephony/providers/AGENTS.md`, its provider registry, `ari/` and `cloudonix/` providers, deployment/dependency docs, model-provider adapters, usage/quotas, and its **current** pinned release.
- Diago: https://github.com/emiago/diago — Go SIP/RTP engine, MPL-2.0. Existing sidecar specifies Diago `v0.41.0`, sipgo `v1.6.0`, Go `1.25.0`; independently verify these APIs and toolchain before changing versions.
- Android gateway: https://github.com/sparkcyf/android-voice-sip-agent — documented one-line gateway, request-URI number or `X-VSA-Dial` header, Direct telephony, GPL-2.0-or-later. Test the existing `android-gw` direct-LAN behavior instead of re-provisioning the handset.
- Pipecat: https://github.com/pipecat-ai/pipecat — established audio pipeline, VAD/turn management/transports. Its WebSocket transports can connect media from a server-side bridge. Prefer a supported provider extension to a Dograh core fork.
- LiveKit: https://github.com/livekit/agents and https://github.com/livekit/sip — serious second option, but a self-hosted SIP server also requires LiveKit server/Redis and can duplicate existing telephony. Do NOT add unless a measured gap requires it.
- `engasix/baat-bot`: https://github.com/engasix/baat-bot — Urdu voice-agent reference only; audit license before code reuse, and be aware of Asterisk-centric design.
- Local ASR/TTS/LLM candidates: faster-whisper, Silero VAD or the chosen framework's turn detector, Piper Urdu Fasih voice, and Ollama/Qwen or another locally run Urdu-capable model. **Benchmark on the actual laptop**; validate each exact engine/model/voice license and quality before committing.
- Candidate Windows production PBX (only when needed): https://github.com/yatevoip/yate — GPL-2.0, interoperability and maintenance unverified. Evaluate rather than assume.

## Decision order — keep this narrow and evidence based
### Gate 0: inventory and preserve
- Inventory `D:\calling` and its Git status, actual active versions, program paths, available disk/RAM/CPU/GPU, installed Go, Python, Docker/WSL (CHECK only), services/port bindings. Never overwrite the existing source or Asterisk/VM files; back up miniSIPServer and MicroSIP configuration **without logging secrets**.
- Review the generated Go bridge and Python adapter; run `go test ./...` and `go build`, plus `pytest`, lint and type checks where configured. Capture exact errors and fix as necessary. Keep this bridge as a candidate, not the finished platform.
- Test offline call control and audio loopback using simulated SIP peers and deterministic audio fixtures. Confirm Go handles G.711 codecs and 20ms PCM16 frames in the right sample rate.

### Gate 1: shortlist *one* existing AI engine; do not code two
- First inspect **Dograh's custom telephony-provider registry**, its documented extension contract, Pipecat serializers and local/private transport. Its provider guide says variations belong within a self-registering package under `api/services/telephony/providers/<name>/`, with minimal import/schema registration outside. Confirm against the selected commit.
- Measure whether Dograh's local deployment is practically possible on the current Windows host without unacceptable disk/runtime overhead. Test Dograh's text/audio browser flow on LOCAL models where supported. A claim of 'no API key needed' is NOT proof of unlimited offline inference: verify actual traffic and whether any hosted model is used.
- If feasible, use Dograh as the sole conversation/workflow engine, implement a minimal **`nrsp_sip_gateway` telephony provider** that adapts the existing Go/Diago sidecar's events and bidirectional PCM. No changes to core pipeline unless unavoidable and documented.
- If Dograh is infeasible, run **Pipecat as the voice-engine framework** with the existing Go media sidecar, keeping call-control/workflow contracts stable. No new GUI or custom telephony framework; keep only the NRSP workflow. Record reason with reproducible evidence in `DECISIONS.md`.
- Do not install/deploy LiveKit, Asterisk, Yate or other infrastructure concurrently as a 'just in case'.

### Gate 2: complete and test the *existing* Go SIP/RTP adapter
- Preserve `nrsp_agent/telephony/contracts.py` as a portability seam; align it with the chosen framework's event/audio types without duplicating its whole abstractions.
- Register a **new distinct test SIP account**, e.g. extension `101` (configured by the operator), not MicroSIP's `100`, using a unique bind port. Do not compete for UDP 5060 or use MicroSIP account simultaneously from two endpoints.
- Confirm sidecar can originate a call to miniSIPServer with the **PBX prefix `9`**, which PBX strips, and that ZTE dials the correct `03...` number. `9` is a PBX routing prefix, not Pakistan's `+92` dial code.
- Verify INVITE/ringing, answer only once the cellular recipient answers, RTP reception (telephone → engine), PCM injection (engine → phone), hangup from either side, failed/rejected calls, timeouts, and reconnect behavior. Test using an explicit allowlisted number and a manual live-call gate.
- Diago sidecar currently uses NDJSON+Base64 across loopback (`127.0.0.1:8765`) for PCM. **Do not rewrite without a benchmark.** At 8kHz mono PCM16, raw 20ms frames are 320 bytes; a few channels are not inherently a large IPC throughput burden. Record CPU, memory, queue depth and audio delay under 1 / 3 / 5 / 10 simulated concurrent sessions; if serialization/jitter is a real bottleneck, use binary audio frames over a proven transport while retaining existing call APIs.
- Examine frame alignment (Go `Read` can return partial buffers), resampling 8/16/24 kHz, packet pacing, jitter, cancellations, dangling goroutines, race conditions, closed queues, backpressure, SIP transaction timeouts, re-registration, call ID isolation and lifecycle idempotency. Fix issues with tests.

### Gate 3: working Urdu agent built on an established engine
- Set up pipeline using supported built-in components rather than handwritten loops: microphone/telephony PCM → VAD + streaming/turn-based Urdu ASR → NRSP state machine + local Urdu-capable LLM → local Urdu TTS → PCM → Go SIP bridge.
- Candidate baseline: faster-whisper with an empirically selected model, local Ollama/Qwen where it actually handles Urdu, Piper Urdu Fasih voice after a pronunciation/commercial-license test; change models based on metrics. Use existing Dograh built-ins when they meet true zero-fee, local, commercial-licensing constraints.
- Implement small, stable NRSP workflow: identify caller as an AI assistant; ask for language preference; get consent/continue; confirm identity **before** revealing account data; ask only authorized loan-verification questions; clarify, repeat and escalate; respect opt-out; record structured outcome; safely end the call. Make numeric/date/amount speech in Pakistani Urdu comprehensible. Keep prompts versioned; use mock borrower data initially.
- Support barge-in, silence, replay, background noise, echo prevention, voicemail/no-answer, voice turn boundaries and human escalation according to actual capabilities. Ensure cancellations interrupt in-flight TTS appropriately. Do not claim full-duplex interruption before testing.

### Gate 4: minimal durable operations (same deployment code)
- Start with SQLite, transactions, idempotent CSV import, explicit campaign authorization, per-SIM capacity=1, retries with backoff and maximum limits, opt-outs, quiet hours, statuses, report export and crash reconciliation. No Redis/Kafka/Postgres/K8s unless measured demand justifies.
- Separate `GatewayRegistry` / `CallSession` / `AgentEngine` / `CustomerStore` so adding phones is configuration + capacity only; do NOT duplicate agent processes per SIM if one worker can manage sessions correctly.
- Maintain a simple Windows startup script, pinned deps, config validation, locally stored/restricted secrets, structured rotating logs, graceful shutdown, service recovery, secure LAN binding, and a `doctor` diagnostic command. Native Windows service only if required for unattended operation.
- Simulate 5 and 10 concurrent channels before testing additional real SIMs. For real concurrent calls, each physical SIM/gateway is only one channel; do not invent throughput numbers.

### Gate 5: production deployment verification, not a post-demo rebuild
- Audit all commercial model/voice/code licenses and the carrier's gateway usage rules before real customer rollout. miniSIPServer free edition is a TEST dependency; document a PBX replacement/upgrade validation plan that keeps `TelephonyPort` stable. Commercially usable OSS can be lawful without separately contacting developers if license terms are satisfied; do not assume all GitHub code is licensed for reuse.
- Prove stability across reboot/network loss, Shizuku restart, registration expiry, call drop, model failure, retry and disk/DB recovery; test secrets redaction and call metadata privacy.
- Success means **the same core agent used for one verified Urdu call can handle 5–10 gateways after adding physical channels and measured compute**, without rewriting the conversational engine.

## Quantitative performance tests (targets, not promises)
Report per call: time to first greeting after answer, ASR end-of-turn latency, LLM first token, TTS first playable audio, full response latency p50/p95, VAD false triggers, interruption latency, UDP RTP packet loss/jitter, STT word/error review on Urdu sample calls, CPU/GPU peak, RAM, disk, and failed-call codes. Begin with working SLA targets (e.g. first spoken reply <=2s p50, <=3.5s p95 under a 1-call local test) and refine against measured quality; do not hide failures behind averages. Run synthetic 1/3/5/10 sessions separately from actual one-SIM testing.

## Existing source inspection priorities (do not bypass)
1. `transport/diago-sidecar/bridge.go`: Go SIP registration/INVITE flow, `setupPCM`, reader/writer, framing, queue bounds, congestion/backpressure, cancellation, dialog lifetime.
2. `transport/diago-sidecar/protocol.go`: NDJSON command/event/audio serialization, per-call authentication and control channel.
3. `transport/diago-sidecar/config.go`: one-SIM call limit, live-call opt-in, allowlist, localhost control.
4. `nrsp_agent/telephony/diago.py`: asyncio read loop, queue overflow handling, closing streams on failures, timeout/reconnect semantics.
5. `tests/test_diago_adapter.py`: FakeBridge tests are **mock coverage only**. Write SIP/RTP integration tests and conduct the explicitly authorized hardware test later.

## Required deliverables IN THIS EXISTING WORKSPACE
- `README.md`: concise Windows setup and architecture, real commands to run and diagnose.
- `DECISIONS.md`: evidence-based platform selection, actual local dependencies, licensing checklist, alternatives rejected.
- `STATUS.md`: verified vs mock vs blocked, with timestamps and measured output (no invented claims).
- `docs/architecture.md`: one diagram, ports, processes, data flow, gateway capacity, failure handling.
- `config/.env.example`: safe defaults, no embedded secrets or real numbers, live calls OFF by default.
- Runnable setup/doctor scripts in PowerShell, pinned dependency manifests/lockfiles, small repeatable smoke tests.
- Reused/extended Go bridge and a narrowly scoped conversation-engine integration (Dograh custom provider if verified feasible; else Pipecat adapter).
- Versioned sample NRSP Urdu verification dialogue with **fake test records** and privacy/opt-out safeguards.
- Benchmarks and a short release checklist including license/production PBX gate.

## Execution protocol
1. **Start NOW by inspecting files already in `D:\calling`, not by generating another scaffolding tree.** Preserve the 13-file change set; if uncommitted, present or create a recoverable backup/commit without overwriting operator changes. Find and inspect the earlier ZIP if present, but do not require it if sources exist.
2. Research Dograh integration/installation and inspect source for exact transport contracts; inspect current Diago code; make ONE short decision (Dograh or Pipecat fallback) with specific reasons and actual Windows prerequisites. Prefer running useful tests immediately over producing long proposals.
3. Fix failing tests and Go build; prove offline two-way SIP/RTP media with a simulated peer. Then integrate chosen OSS agent's existing pipeline. After allowlisted, manually authorized call, verify two-way **AI** audio through the real phone.
4. Add NRSP workflow and durable queue only after actual speech loop works, but keep extension points from the beginning.
5. At each gate: run tests, provide exact terminal commands, evidence, changed-file summary, remaining blockers and next step. Do not ask the operator to repeat already known setup facts. Do not launch real calls, install paid tools, expose public ports, disable Windows firewall, or replace miniSIPServer without explicit approval.
6. **Do not stop at documentation. Implement as much working code and automated tests as the local environment permits.** If a missing permission, model file, OS dependency or API truly blocks progress, implement the safe offline path, report the single required operator action and continue other independent work.

## Why this choice
- **Keep** tested Windows → ZTE/Jazz telephony and existing Go RTP code.
- **Reuse** Dograh or Pipecat as established conversational voice software.
- **Only custom-connect** the two systems and encode the NRSP business workflow.
- **Benchmark** before rewriting for speed.
- **Harden** exactly the same code for commercial production, without overengineering the initial deployment.

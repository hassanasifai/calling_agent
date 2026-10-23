# Architecture decisions

## ADR-001: Pipecat is the embedded conversation engine

Status: accepted for the Windows-native implementation, 2026-10-23.

Dograh `a9b1d831` (release line 1.49) was inspected, including its telephony
provider registry and ARI provider. Its provider seam is sound and could wrap
this project's Pipecat transport. Its supported deployment, however, is a
multi-service Docker stack containing PostgreSQL, Redis, MinIO, API/worker/UI
services and optional coturn. Redis Open Source is available on Windows only
through Docker; the native supported alternative is a separate commercial
product. Therefore full Dograh is not a supportable native-Windows Server
dependency under the no-Docker/WSL and zero-service-fee constraints. This is
independent of laptop capacity.

Pipecat 1.13.0 installs from native Python 3.13 Windows wheels and supplies the
needed mature functions directly: frame pipelines, local Whisper/Piper/Ollama
services, Silero VAD, turn aggregation, interruptions, metrics and worker
lifecycle. The custom integration is limited to a `BaseTransport` adapter over
the preserved `TelephonyPort` contract.

Dograh remains a reviewed future control-plane option if its supported service
stack is approved. The adapter deliberately uses Pipecat's public transport API
so it can also be instantiated by a Dograh custom provider without replacing
Go RTP code.

## ADR-002: Do not insert LiveKit in the Jazz media path

Status: accepted for the initial production path, 2026-10-23.

LiveKit `a390d03d` supports a native Windows server binary and is a mature
option for WebRTC rooms and distributed participants. It is not needed between
the local PBX and one Pipecat worker: doing so adds another RTP/WebRTC media hop,
server, credentials and failure domain while duplicating the proven Diago
bridge. LiveKit's source also reports that capacity management is unavailable
on Windows. Keep it available for a future browser supervisor or multi-party
requirement, not as mandatory telephony transport.

## ADR-003: Local model policy

Status: accepted with benchmark and license gates, 2026-10-23.

- ASR default: multilingual Faster-Whisper small, MIT conversion of Whisper.
- TTS default: Piper `ur_PK-fasih-medium`, MIT voice weights; Piper runtime is
  GPL-3.0 and its distribution obligations must be honored.
- LLM demo default: Qwen3 1.7B through Ollama; server starting profile: Qwen3
  8B. Qwen3 is Apache-2.0 and explicitly includes Urdu in its language list.
- Technically stronger restricted-license models may be benchmarked. Production
  use requires recorded license approval, as authorized by the project owner.
- No hosted or metered inference API is permitted.

Model choice is provisional until Urdu accuracy, first-token latency and
concurrent-call benchmarks run on the target Windows Server.

## ADR-004: Deterministic call policy; the LLM only acknowledges

Status: superseded by ADR-007/008 (2026-10-23); kept for its evidence.

Identity confirmation, CNIC-suffix verification, disclosure of amount and due
date, opt-out, wrong person, human handoff, payment difficulty, silence and
closing are decided in code (`nrsp_agent/conversation/policy.py`) and spoken
from fixed templates (`script.py`). The LLM is reached only after
verification, for free-form acknowledgements, and receives a fresh minimal
context (instruction, one example, the caller's last turn) — never earlier
turns that contain customer facts.

Evidence (qwen3:1.7b via Ollama): with an Urdu instruction plus full history
(~680 prompt tokens) the model repeated the amount/date and invented an
"online payment facility"; with the minimal English instruction (~280 tokens,
0.25–0.5 s first token) replies were clean, but for "can I get more time?" it
promised an extension. Hence payment-difficulty turns are scripted, and an
output guard rejects sentences with digits, money words, commitment words
(مہلت، رعایت، معاف…), non-Urdu script, or echoes of the caller's words.

Verification accepts the four expected digits contiguously within at most six
heard digits. Whisper was measured adding a digit from the question
("آخری چار") and repeating the final digit; the six-digit cap stops a caller
reciting 0–9 to cover every window (≤3 windows per attempt, 2 attempts).

## ADR-005: ASR configuration is evidence-driven and conservative

Status: accepted, 2026-10-23.

- Whisper prompts contain no intent trigger words. Prompts with "اللہ حافظ"
  and "نمائندہ" were measured making Whisper hallucinate those words, which
  ended calls as goodbye/handoff. Repetition loops ("X، X، X، X") are treated
  as unclear, never as intents.
- A digits-only prompt is applied only while the CNIC suffix is requested:
  after G.711 A-law it turned "چارڈ، ماد، چیساہر" into "چار، پانچ، چھ، سات";
  applied elsewhere it makes Whisper invent digits. It never contains the
  customer's own digits.
- Decoding is bounded (temperature 0, no conditioning, 96-token cap). The
  faster-whisper defaults looped and re-decoded up to six times: ~40 s per
  segment on CPU; bounded decoding measured 1.9–2.9 s.
- VAD `stop_secs` 0.5 (0.3 split phrases at comma pauses; every fragment pays
  Whisper's fixed ~2 s encoder cost) and turn-stop fallback 20 s (Pipecat's
  5 s default attached late transcripts to the next turn).
- `large-v3-turbo` on the laptop CPU took 6.6–7 s per segment and, with the
  digit prompt, produced a wrong CNIC suffix; it remains the server/GPU
  candidate pending a real-speech accuracy benchmark.

## ADR-006: Media timing owned by Go, paced end to end

Status: accepted, 2026-10-23.

- The Go bridge sends a continuous 20 ms RTP stream (silence when idle) so
  RTP timestamps track wall clock across pauses; when the process misses
  ticks the timestamp skips the gap instead of drifting (measured 4.9 s drift
  over 218 s on the paging laptop before; ≤5 ms after).
- Python paces TTS to the playout clock with an 80 ms lead (Pipecat's base
  output transport does not pace; Piper runs ~18× real time and overflowed
  the bridge queue). Barge-in sends `clear_audio`; the closing line uses a
  drained hangup so the goodbye is not cut.
- The bridge opts out of Windows power throttling and runs at above-normal
  priority; it fails a call after `NRSP_RTP_TIMEOUT_SECONDS` without inbound
  RTP, and reads each decoded RTP packet regardless of the remote ptime.

## ADR-007: Generic profile-driven engine; NRSP is a profile

Status: accepted, 2026-10-23. Supersedes the NRSP-specific policy of ADR-004.

Task behaviour (persona, objective, rules, knowledge, actions, approved texts,
result fields, ASR prompts, voice) lives in `config/profiles/<name>/`. The
engine (`nrsp_agent/agent/`) knows four action kinds — say, record, verify,
end — plus flags that gate availability, confirmations and safety actions. A
second unrelated profile runs on the same code in tests. No exact-phrase
matching decides intent; the LLM chooses an action, code validates it.

## ADR-008: One streamed, schema-constrained LLM call per turn

Status: accepted, 2026-10-23.

The model returns `{"action", "value", "say"}` under a JSON schema built per
turn from the actions allowed now (llama.cpp and Ollama both enforce it by
grammar). Approved-text actions must have an empty `say`, so generation stops
early; the action is applied as soon as its name is complete. Evidence:
per-turn schema + few-shot raised Gemma-4-E2B dev accuracy 65 % → 75 %
(held-out 82 %) and removed invalid outputs. Private call data never reaches
the model; history stores the model's own output. Measured: the ~1,240-token
system prompt costs ~35 s cold on 4–5 CPU threads, so it is identical across
calls (per-call details last), the full call history is kept (a sliding window
caused 300–430-token re-processing, +7–11 s), and warm-ups send the exact
request shape (`chat_template_kwargs` changes the rendered prompt).

## ADR-009: Model selection (CPU, measured)

Status: accepted for Ubuntu dev, 2026-10-23; re-measure on target hardware.

- ASR: whisper-small-urdu (Apache-2.0) as CTranslate2 int8, beam 5: best
  ASR→LLM understanding (58/68 vs 50 base-urdu, 46 Canopy-M); FLEURS WER 28 %
  vs 44 % for multilingual small. Canopy-M is 20× faster but loses 12 points.
- LLM: Gemma-4-E2B-it QAT Q4_0 (Apache-2.0): held-out 82 %, stays within
  rules. Qwen3.5-2B ties on actions but promised "more time" and claimed to be
  human; Qwen3-1.7B fell to 64 % on held-out phrasings.
- TTS: Piper `ur_PK-aegis_female-medium` (MIT): G.711 intelligibility CER 3.8 %
  vs 8.7 % for fasih at equal cost. Kani-TTS-Urdu and Orpheus-Urdu are not
  real-time on this CPU.
- Runtime: official llama.cpp CPU build (runtime AVX-512 kernels), 4.3× faster
  generation than a generic OpenBLAS build.

## ADR-010: Verification inputs and confirmations

Status: accepted, 2026-10-23.

Spoken digits are parsed by code (Urdu, English, Roman Urdu number words) and
read back for yes/no confirmation; the LLM never reconstructs numbers. DTMF
(RFC 2833) keypad entry is offered and compared exactly. Irreversible endings
(opt-out, handoff, wrong person) are confirmed first because the model was
measured choosing them for "who is calling?" and "I can't hear you"; during a
confirmation only yes/no, a new verification attempt and safety exits are
allowed.

## ADR-011: Inference on the VM behind one private API; thin Windows client

Status: accepted, 2026-10-23.

The laptop keeps SIP/RTP (Go/Diago), Pipecat orchestration and Silero VAD;
ASR, LLM and TTS run on the VM as user systemd units (`calling-ai.slice`:
6 CPU, 12 GB, Nice 5, restart always). One aiohttp process
(`nrsp_agent/inference_api.py`) serves ASR and TTS with the same model code as
local mode and proxies the loopback llama-server, so there is one endpoint,
one token and one allowlist. Pipecat's `PiperHttpTTSService` and the existing
OpenAI-compatible LLM client are reused; only ASR needed a thin subclass (to
keep the engine's per-turn Whisper prompt). Binding is to the VPN-reachable
private address only; remote URLs must be private IP literals. Measured on the
VM: remote mode added no measurable latency (ASR round trip 2.301 s vs 2.299 s
server decode); real VPN overhead is to be measured from Windows. Local mode
remains the default and unchanged.

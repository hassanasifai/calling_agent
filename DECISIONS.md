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

Status: accepted, 2026-10-23.

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

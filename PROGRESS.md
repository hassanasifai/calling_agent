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

## Currently working on — reliability + latency pass (2026-10-23, later)
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

## Remaining tasks (priority)
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
- Ollama not installed on this VM (no services allowed); llama.cpp server on
  loopback port 18431 via `scripts/run_llm_server.sh` instead.

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

## Latest test results
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
Windows integration check (Ollama model tag, DTMF through the ZTE gateway) and
a real-speech accuracy set; both need resources outside this VM.

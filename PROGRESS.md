# PROGRESS

Updated: 2026-10-23 (Ubuntu AIUAT dev VM, CPU only, cgroup 6 CPU / 32 GB; Financial AI has priority)

## Overall objective
A reusable, configurable AI voice calling engine (Go/Diago SIP/RTP + Pipecat +
Silero VAD + local Urdu ASR + local LLM + Piper TTS). A task is defined by an
agent profile (prompt/persona, knowledge, objective, actions, rules, result
fields); the core pipeline does not change per task. NRSP installment/identity
verification is the first profile and the benchmark.

## Completed and verified
- Ubuntu dev env: `.venv` (Py 3.13, project deps), `.cache/venv-bench`
  (CPU torch, transformers, canopy, CT2). All caches/models inside repo (ignored).
- `scripts/pressure_check.sh` gates every heavy step (MemAvailable, swap I/O, PSI).
- ASR eval sets (`scripts/build_asr_eval.py`): FLEURS ur_pk human read speech
  (100 utts), synthetic domain set (42 utts × 3 voices), 8 noise clips, all
  through 8 kHz G.711 A-law.
- ASR benchmark complete (8 configs, table below).
- Existing 83 unit tests pass on Ubuntu (1 Windows-only skip).

- Generic profile-driven engine (`nrsp_agent/agent/`): profile schema/loader,
  gated actions (say/record/verify/end), verification by code-parsed digits with
  read-back, BM25 knowledge, structured result, per-turn JSON-schema-constrained
  LLM output (only allowed actions), few-shot examples in the cached prompt.
  NRSP is `config/profiles/nrsp_verification/`; a second unrelated profile
  (`tests/fixtures/profiles/clinic_reminder/`) runs on the same engine.
  Unit-tested (71 pass); not yet validated end-to-end over SIP on Ubuntu.
- Official llama.cpp b11516 CPU build in `.cache/llama.cpp` (AVX-512 kernels):
  4.3× faster generation than the shared OpenBLAS binary (13.9 vs 3.2 tok/s).
- Go 1.26.9 in `.cache/go-dist` (user space) for the bridge/loopback.

## Currently working on
- LLM selection on the dev turn set with the final engine prompt/schema:
  Gemma-4-E2B-it (QAT Q4_0), Qwen3.5-2B, Qwen3-1.7B.

## Remaining tasks (priority order)
1. Generic engine: profile schema/loader, action executor, BM25 knowledge, prompt builder; NRSP profile; second test profile.
2. Tests: profile switching, gated actions, unfamiliar multi-turn dialogues (scripted + real LLM; dev vs held-out).
3. End-to-end understanding eval: ASR hypotheses → LLM action → outcome, for ASR finalists.
4. TTS: Piper fasih vs aegis_female (latency, G.711 intelligibility); Kani-Urdu CPU feasibility.
5. Go toolchain (user-space) → loopback SIP/RTP E2E on Ubuntu with llama-server; measure end-of-speech→first-audio.
6. Update STATUS.md / DECISIONS.md; commit.

## Issues and blockers
- No real human Pakistani *telephone call* recordings available; FLEURS is read speech, domain set is synthetic. Field accuracy unverified.
- Host CPU shared with busy Financial AI llama-servers (load 6–15); latencies are under contention.
- Isolated spoken digits over G.711 are poorly recognised by all ASR candidates (≤7/24 lexically parsed).
- Ollama not installed here; llama.cpp `llama-server` (existing user binary, own port/weights) used instead. Windows path keeps Ollama (same OpenAI-compatible API).

## Benchmarks
### ASR (4 threads, G.711; domain = synthetic, FLEURS = human read)
| Model | Domain P50/P95 s | Domain WER/CER | FLEURS WER/CER | FLEURS P50 s | Digits | Noise halluc. | RSS MB |
|---|---|---|---|---|---|---|---|
| Canopy-M (Conformer-CTC 58M) | 0.12/0.15 | 40.9/16.3 | 49.3/20.9 | 0.27 | 0/24 | 8/8 (1-char) | 727 |
| Whisper-small (baseline) | 2.56/3.04 | 32.3/14.5 | 43.8/20.1 | 3.75 | 6/24 | 0/8 | 1006 |
| whisper-small-urdu (CT2 int8) | 2.66/3.76 | 24.5/10.4 | 28.3/11.5 | 4.33 | 7/24 | 8/8 | 557 |
| whisper-base-urdu-full | 0.99/1.15 | 37.2/14.8 | 43.2/18.0 | 1.55 | 1/24 | 7/8 | 275 |
| whisper-tiny-urdu-merged | 0.61/0.73 | 45.9/20.4 | 60.7/27.3 | 0.98 | 0/24 | 3/8 | 242 |
Without prompts the fine-tunes lose all digits (0/24). Baseline Whisper with the
digit prompt turned "four five six seven" into "5 6 7 8" (prompt-induced sequence).

### LLM (dev set, 40 turns, production prompt; 4 threads, contended host)
Raw speed (llama-bench b11516, 4 thr): Qwen2.5-1.5B pp512 87 / tg 13.9 tok/s;
Qwen3-1.7B 83 / 13.2; Qwen3-0.6B 135 / 21.
| Model | Action acc | False call-ending | Action decided P50 | Notes |
|---|---|---|---|---|
| Qwen2.5-1.5B (first engine prompt) | 47.5% | 16/40 | 2.1 s | poor Urdu replies |
| Qwen3-1.7B (first engine prompt) | 45% | 6/40 | 2.6 s | hallucinated amounts/dates (guard caught) |
| Gemma-4-E2B (first engine prompt) | 65% | 5/40 | 3.7 s | most natural Urdu |
Rerun with per-turn schema + few-shot examples in progress.
Rejected for CPU real time: Orpheus-Urdu 3B (author: "not for production"), see TTS.

### TTS
Pending (Piper fasih vs new Piper aegis_female; Kani-Urdu token-rate check).

## Architecture decisions
- Keep Go/Diago + Pipecat + Silero; no Dograh/LiveKit/new services.
- LLM-first understanding; code validates consequential actions; sensitive
  numbers are parsed from the transcript by code, never reconstructed by the LLM.
- Canopy loaded with `torch.load(weights_only=True)` (upstream uses unsafe pickle).

## Latest test results
91 passed, 7 skipped (interim code). Lint clean.

## Immediate next action
Finish LLM comparison; implement generic profile engine.

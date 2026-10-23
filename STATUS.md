# Implementation status

Updated: 2026-10-23 (Ubuntu AIUAT dev VM, CPU only, shared with Financial AI)

## Milestone: generic profile-driven calling engine — offline SIP/RTP validated

The engine is task-agnostic: a call is driven by an **agent profile**
(`config/profiles/<name>/profile.toml` + knowledge files). NRSP installment
reminder with identity verification is the first profile; an unrelated clinic
profile runs on the same code in tests.

Stack: Go/Diago bridge ⇄ SIP/RTP (G.711) ⇄ Pipecat 1.13 (Silero VAD,
whisper-small-urdu via faster-whisper, Gemma-4-E2B via llama.cpp/Ollama
OpenAI API, Piper `ur_PK-aegis_female-medium`) ⇄ generic `CallEngine`.

### Offline SIP/RTP loopback (real bridge, test PBX/callee, real models)
`python -m nrsp_agent.devtools.loopback all --profile config/ubuntu-dev.toml`
(needs `scripts/run_llm_server.sh`). Final run, 9 scenarios:

| Scenario | Result |
|---|---|
| happy_path | completed, verified (spoken digits + read-back), promise recorded |
| barge_in | completed; bot audio stopped 600 ms after caller onset |
| wrong_person / handoff | correct outcome after the profile's confirmation question |
| silence / remote_hangup_mid_turn / rtp_loss | correct teardown |
| keypad_verification | DTMF path verified in earlier run; final run: caller's garbled "جی ہاں" read as wrong person (understanding miss) |
| opt_out | passed in 2 of 3 runs; final run: caller's short "جی ہاں" never produced a turn (see open issues) |

Media: 0 sequence gaps, 1 SSRC, jitter ≤ 1.6 ms, drift ≤ 41 ms in every call;
RTP timestamp skips (8–73 per call) coincide with host load (max inter-arrival
79 ms) — the bridge's designed tick-skip, not drift.

Latency (peer-measured, 18 caller turns): **end of caller speech → first bot
audio P50 8.1 s, P95 11.7 s** on this contended CPU. Answer → greeting 0.9 s.
Per stage: ASR 2.65 s P50, LLM first token 2.3 s P50 (+~0.8 s to the action),
TTS first audio 0.54 s P50, VAD end-of-speech 0.5 s. **The <2 s target is not
met on CPU**; ASR + LLM are ~85 % of the path and both run ~10× faster on a GPU
(endpoints are configurable, no code change).

### Verification
Python: 78 passed, 9 skipped (opt-in E2E + Windows-only); ruff/black clean.
Go: `go test ./...` passes (bridge + media integration, DTMF reader wiring).
Benchmarks and evaluation sets: see PROGRESS.md.

## Changes this milestone
- Generic engine, NRSP moved to a profile; LLM-first understanding with
  per-turn JSON-schema-constrained actions; code-validated consequential steps.
- Model selection by measurement: whisper-small-urdu (ASR), Gemma-4-E2B (LLM),
  Piper aegis_female (TTS). See PROGRESS.md for every rejected candidate.
- Verification: code-parsed spoken digits with read-back; DTMF keypad entry.
- Confirmation before irreversible endings (opt-out, handoff, wrong person).
- Prefix-cache-friendly prompts (cross-call static system prompt, full in-call
  history, warm-ups with identical request shape); stale-answer protection.
- VAD tuned for short Urdu answers (confidence 0.5, start 0.1 s).

## Not verified / open issues
- **No real call.** miniSIPServer/ZTE/Jazz untested with this build; whether the
  ZTE gateway forwards RFC 2833 DTMF is unknown.
- **Accuracy on real Pakistani callers is unmeasured**: no recorded telephone
  speech; FLEURS is read speech; domain and loopback callers are synthetic.
- Spoken CNIC digits are unreliable on CPU ASR (≤25 % per attempt on non-bot
  voices); keypad entry is the recommended path.
- Intermittent: a short caller answer right after a bot question sometimes
  produces no VAD event live (same audio replayed offline is detected). Impact:
  one silence re-prompt. Next step: per-frame VAD logging on the live path.
- LLM understanding errors on garbled ASR (e.g. "کی ہند میں احمد علی…").
- Windows: Ollama tag for Gemma-4-E2B and Windows latency not yet measured.
- Naturalness of the aegis voice needs human listening
  (`data/eval/tts_samples/`).

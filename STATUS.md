# Implementation status

Updated: 2026-10-23

## Milestone: offline end-to-end Urdu conversation — PASS (synthetic caller)

A complete Urdu AI conversation runs through the real components: Go/Diago
bridge ⇄ real SIP (digest REGISTER + 407 INVITE) and G.711 PCMA RTP ⇄ a
loopback test PBX/callee (`transport/diago-sidecar/testpeer`) that plays
telephone-band Urdu caller speech and audits every RTP packet ⇄ Python
`DiagoTelephonyPort` ⇄ Pipecat 1.13 (Silero VAD, Faster-Whisper small,
Qwen3 1.7B via Ollama, Piper ur_PK-fasih) ⇄ deterministic NRSP call policy.

Run: `python -m nrsp_agent.devtools.loopback all` (see README).

Measured on the demo laptop (i7-13620H, 7.7 GB, CPU only), after freeing RAM,
PCMA, 8 scenarios:

| Scenario | Outcome | Notes |
|---|---|---|
| happy_path | completed, verified | greeting → identity → CNIC suffix → notice → LLM reply → goodbye → drained BYE |
| barge_in | completed | caller interrupts greeting; bot audio stopped 700 ms after caller onset |
| opt_out / wrong_person | opted_out / wrong_person | no amount/date disclosed |
| silence | no_response | two prompts, then close |
| remote_hangup_mid_turn | remote_hangup | BYE during ASR; clean teardown |
| rtp_loss | rtp_timeout | bridge emits RTP_LOST, sends BYE |
| handoff | **missed** (fails safe) | Whisper heard "نمائندے" as "نمائن دیس"; re-asked, then closed |

Latency (happy path; peer-measured, caller speech end → bot speech start):
2.4–3.3 s on scripted turns, 6.9 s on the LLM turn. Answer → first bot audio
220–400 ms. Per-service: Whisper small 1.8–2.8 s per turn (dominant; fixed
~2 s encoder cost), Qwen3 first token 1.25 s / full reply 4.4 s, Piper first
audio 0.04–0.15 s. RTP: 0 sequence/timestamp errors, 1 SSRC, jitter ≤0.5 ms,
clock drift ≤0.6 ms per call; 0 dropped caller frames.

Verification: Go `go test ./...` (loopback two-way PCM per codec; full bridge
over a digest-auth PBX: events, RTP integrity and drift, one call per SIM,
clear_audio barge-in, drained hangup, RTP-loss watchdog). Python: 84 unit
tests (policy, guard, controller, transport pacing/clear, session ordering
and teardown, adapter), ruff + black clean. Opt-in E2E:
`NRSP_E2E=1 pytest tests/e2e`.

## Defects found and fixed this milestone

- Qwen3 over Ollama `/v1` spent the whole budget on hidden reasoning and
  returned no content → `reasoning_effort: none` (first token 0.13 s warm).
- Pipecat output was unpaced; Piper (~18× real time) overflowed the bridge
  queue and truncated replies → paced output, 80 ms lead.
- No far-end clear on barge-in → `clear_audio` command.
- RTP sent only while speaking, timestamps drifted; missed ticks drifted
  permanently (4.9 s/218 s under paging) → continuous clock with re-align.
- No inbound-RTP watchdog; fixed 20 ms read assumed remote ptime.
- Audio could be sent before `RTP_READY`; models loaded after answer; greeting
  LLM-generated; deprecated turn filter; Whisper ignored the project model
  cache; full audio queue tore down the shared bridge connection.
- Slow ASR attached transcripts to the next turn (5 s turn-stop default).
- Whisper prompt words hallucinated into intents (goodbye, handoff) and
  ended calls; repetition loops took ~40 s per segment → see ADR-005.
- Silent TTS contexts could mark TTS unusable and end the call.

## Not verified / open gates

- **No real call yet.** miniSIPServer/ZTE/Jazz with this build is untested;
  `docs/jazz-test-runbook.md` is prepared. Requires explicit authorization of
  one exact test number.
- **ASR accuracy is the main risk.** Fixtures are synthetic Piper speech;
  Whisper-small over G.711 garbles many words (intents still matched 7/8
  scenarios). Needs a real recorded Urdu telephone evaluation set; compare
  small vs. large-v3-turbo on the Windows Server (turbo took 6.6–7 s/segment
  on the laptop CPU).
- LLM quality with qwen3:1.7b is acceptable only because the policy and guard
  constrain it; qwen3:8b (server profile) is unbenchmarked.
- Demo laptop needs ~3.5 GB free RAM; with browsers/WSL open it paged and the
  LLM first token reached 33 s. CPU throttling observed (perf 49–127%).
- Barge-in during the closing line, digits in non-verification turns while
  the digit ASR prompt is active, and µ-law (PCMU) end-to-end are untested.
- Durable queue, call records storage, multi-SIM scheduling, Windows service,
  concurrency benchmarks: not started.

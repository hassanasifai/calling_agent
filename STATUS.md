# Implementation status

Updated: 2026-10-23

## Verified now

- Existing Go/Diago and Python work preserved in Git baseline `a65e7fc`.
- Diago pinned at v0.41.0; SIPGo pinned at v1.6.0.
- Go bridge compiles on Windows and all Go tests pass.
- In-process SIP/RTP tests prove answer, bidirectional PCM through PCMA and
  PCMU, and hangup without a telephone call.
- Python bridge adapter, Pipecat transport and answered/hangup lifecycle tests
  pass; 15 tests provide 86.64% total coverage.
- Pipecat 1.13.0 plus local Piper, Faster-Whisper and ONNX dependencies install
  as native Windows Python 3.13 wheels.
- Urdu Piper synthesis ran locally: 3.646 s audio in 0.198 s (RTF 0.054).
- Faster-Whisper small transcribed that synthetic clip in 2.155 s (RTF 0.591).
  The single result was intelligible but inaccurate and is not an accuracy
  acceptance test.
- Qwen3 1.7B produced an appropriate Urdu response locally. Cold response was
  23.087 s including 19.402 s model load; warm response was 3.122 s at 17.69
  generated tokens/s. This proves execution, not conversation quality.
- Research snapshots exist under ignored `_research`: Dograh `a9b1d831`,
  Pipecat `06d3e88b`, Diago `5d7145c6`, ZTE SIP agent `a134b8e5`, and LiveKit
  `a390d03d`.

## Gate status

- Gate A offline protocol/media verification: PASS.
- Gate A real miniSIPServer/ZTE/Jazz call from this build: NOT RUN; explicit
  authorization for an allowlisted number is required.
- Local Urdu TTS/ASR execution: PASS for smoke/performance, NOT YET PASS for
  quality.
- Local Urdu LLM execution: PASS for smoke/performance, NOT YET PASS for
  conversation quality.
- End-to-end Urdu conversation through Pipecat and the loopback SIP peer: next.
- Durable queue, call tracking, resilience and multi-SIM scheduling: not yet
  implemented; they follow the first end-to-end conversation gate.

## Known production gates

- Run a curated Urdu/noisy-telephone evaluation set and choose models from
  measured accuracy and latency.
- Benchmark 1/3/5/10 simultaneous sessions on the target Windows Server.
- Obtain and record approval for any restricted-license model selected over
  the permissive defaults.
- Conduct one explicitly authorized allowlisted hardware call and capture SIP,
  media and pipeline metrics without customer data.

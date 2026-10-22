# Architecture

```text
Jazz SIM ←→ ZTE Voice SIP Agent ←→ miniSIPServer
                                      ↕ SIP + G.711 RTP
                         Go/Diago sidecar (one call per session)
                                      ↕ loopback NDJSON + PCM16
                         Python TelephonyPort / Pipecat transport
                                      ↕ Pipecat frames
                   Silero VAD → Faster-Whisper → Ollama/Qwen3
                                      → Piper Urdu → interruption-aware output
```

The call leg is unchanged from the proven lab setup. Go performs registration,
call state, RTP packetization, PCMA/PCMU conversion and 20 ms playback pacing.
The loopback control connection is authenticated. Python receives PCM16,
resamples caller audio from 8 kHz to 16 kHz for ASR/VAD, and sends 8 kHz PCM16
back to Go. Pipecat provides pipeline lifecycle, turn-taking and interruption
semantics.

Each active call receives an isolated transport and pipeline worker. The later
queue/scheduler may start multiple instances behind separate gateway IDs; it
must not share call media queues. Model services can be colocated for the demo
and moved to larger local Windows hosts without changing the telephony
contract. Concurrency limits are benchmark-derived, never inferred from the
development laptop.

Dograh can become a control plane if its supported dependencies are approved;
the public Pipecat transport remains reusable in its provider factory. LiveKit
is reserved for requirements that genuinely need rooms/WebRTC, such as a web
supervisor, rather than being inserted into the PBX path preemptively.

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

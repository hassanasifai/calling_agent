"""Offline end-to-end conversation harness over real SIP/RTP on loopback.

Starts the real Go bridge and the Go SIP test peer (registrar + scripted
caller) as subprocesses, then runs :func:`run_outbound_session` with the real
local models. The bridge's PBX is forced to the loopback test peer, so no real
telephone network is reachable; the "number" is a fake test extension.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from nrsp_agent.agent import build_engine
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA
from nrsp_agent.session import SessionResult, run_outbound_session
from nrsp_agent.telephony.diago import DiagoTelephonyPort
from nrsp_agent.voice.local_pipeline import (
    LLM_EXTRA_BODY,
    LocalVoicePipelineSettings,
    preload_models,
)

ROOT = Path(__file__).resolve().parents[2]
GO_DIR = ROOT / "transport" / "diago-sidecar"
FIXTURES = ROOT / "tests" / "fixtures" / "caller"
TEST_NUMBER = "9000"


@dataclass(frozen=True)
class Scenario:
    name: str
    utterances: list[dict[str, Any]]
    final_action: str = "wait_for_bye"
    final_delay_ms: int = 0
    rtp_timeout_s: int = 15
    bot_silence_ms: int = 1200
    max_call_s: int = 240


def _u(name: str, mode: str = "after_bot", delay_ms: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "wav": str(FIXTURES / f"{name}.wav"),
        "mode": mode,
        "delay_ms": delay_ms,
    }


def _keys(name: str, digits: str) -> dict[str, Any]:
    """Keypad entry sent as RFC 2833 telephone-events after the bot speaks."""
    return {"name": name, "digits": digits, "mode": "after_bot", "delay_ms": 0}


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            "happy_path",
            [
                _u("confirm_identity"),
                _u("cnic_last4"),
                _u("yes_correct"),
                _u("pay_on_time"),
                _u("goodbye"),
            ],
        ),
        Scenario(
            "keypad_verification",
            [
                _u("confirm_identity"),
                _keys("cnic_keypad", TEST_CALL_DATA["cnic_last4"]),
                _u("pay_on_time"),
                _u("goodbye"),
            ],
        ),
        Scenario(
            "barge_in",
            [
                _u("barge_in_identity", "barge_in", 1800),
                _u("cnic_last4"),
                _u("yes_correct"),
                _u("goodbye"),
            ],
        ),
        # Irreversible endings are confirmed by the profile before the call ends.
        Scenario("opt_out", [_u("opt_out"), _u("yes")]),
        Scenario("wrong_person", [_u("wrong_person"), _u("yes")]),
        Scenario("handoff", [_u("handoff"), _u("yes")]),
        Scenario("silence", []),
        Scenario(
            "remote_hangup_mid_turn",
            [_u("confirm_identity")],
            final_action="hangup_after_delay",
            final_delay_ms=900,
        ),
        Scenario(
            "rtp_loss",
            [_u("confirm_identity")],
            final_action="stop_rtp",
            rtp_timeout_s=4,
        ),
    )
}


@dataclass
class RunResult:
    scenario: str
    session: SessionResult | None
    peer: dict[str, Any]
    out_dir: Path
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        s = self.session
        return {
            "scenario": self.scenario,
            "error": self.error,
            "session": (
                None
                if s is None
                else {
                    "result": s.result,
                    "reason": s.reason,
                    "outcome": s.outcome,
                    "call_result": s.call_result,
                    "guard_rejections": s.guard_rejections,
                }
            ),
            "transcript": [] if s is None else s.transcript,
            "events": [] if s is None else s.events,
            "peer": {k: v for k, v in self.peer.items() if k != "rtp"},
            "rtp": self.peer.get("rtp"),
            "metrics": [] if s is None else s.metrics,
            "timings": self.timings,
        }


def _free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_go_binaries() -> tuple[Path, Path]:
    bin_dir = GO_DIR / "bin"
    bridge = bin_dir / "nrsp-diago-bridge.exe"
    peer = bin_dir / "sip-test-peer.exe"
    go = shutil.which("go") or str(ROOT / ".cache" / "go-dist" / "go" / "bin" / "go")
    for target, package in ((bridge, "."), (peer, "./cmd/sip-test-peer")):
        subprocess.run(
            [go, "build", "-o", str(target), package], cwd=GO_DIR, check=True
        )
    return bridge, peer


def warm_ollama(settings: LocalVoicePipelineSettings) -> float:
    """Load the LLM into memory before the call; returns seconds taken."""
    body = json.dumps(
        {
            "model": settings.llm_model,
            # The profile's system prompt is shared by every call, so warming
            # it once fills the server's prefix cache for all calls.
            "messages": [
                {
                    "role": "system",
                    "content": build_engine(
                        NRSP_PROFILE, TEST_CALL_DATA
                    ).system_prompt(),
                },
                {"role": "user", "content": "."},
            ],
            "max_tokens": 1,
            "reasoning_effort": "none",
            **LLM_EXTRA_BODY,
        }
    ).encode()
    request = urllib.request.Request(
        f"{settings.llm_base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", **settings.auth_headers()},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        response.read()
    return time.perf_counter() - started


def cpu_performance_pct() -> float | None:
    """Windows '% Processor Performance' (100 = base clock; turbo is >100).

    Recorded with every run because thermal throttling on the demo laptop was
    measured to slow Whisper 3.5x; latencies are meaningless without it.
    """
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Counter '\\Processor Information(_Total)"
                "\\% Processor Performance').CounterSamples.CookedValue",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return round(float(out.stdout.strip().splitlines()[-1]), 1)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _prepare_run_dir(
    out_dir: Path, scenario: Scenario
) -> tuple[Path, BinaryIO, BinaryIO]:
    """Write the peer script and open process logs (sync file I/O)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "script.json"
    script_path.write_text(
        json.dumps(
            {
                "bot_silence_ms": scenario.bot_silence_ms,
                "reply_timeout_s": 90,
                "max_call_s": scenario.max_call_s,
                "final_action": scenario.final_action,
                "final_delay_ms": scenario.final_delay_ms,
                "utterances": scenario.utterances,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        script_path,
        open(out_dir / "peer.log", "wb"),
        open(out_dir / "bridge.log", "wb"),
    )


def _write_summary(result: RunResult) -> None:
    (result.out_dir / "summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def _wait_line(stream: asyncio.StreamReader, prefix: str, seconds: float) -> str:
    async def read() -> str:
        while True:
            line = (await stream.readline()).decode(errors="replace")
            if not line:
                raise RuntimeError(f"process exited before printing {prefix}")
            if line.startswith(prefix):
                return line.strip()

    return await asyncio.wait_for(read(), seconds)


async def run_scenario(
    scenario: Scenario,
    out_dir: Path,
    settings: LocalVoicePipelineSettings,
    binaries: tuple[Path, Path],
    *,
    codec: str = "PCMA",
) -> RunResult:
    bridge_bin, peer_bin = binaries
    peer_port = _free_port(socket.SOCK_DGRAM)
    control_port = _free_port(socket.SOCK_STREAM)
    sip_password = secrets.token_urlsafe(12)
    token = secrets.token_urlsafe(24)
    script_path, peer_log, bridge_log = _prepare_run_dir(out_dir, scenario)

    peer = await asyncio.create_subprocess_exec(
        str(peer_bin),
        "-bind",
        "127.0.0.1",
        "-port",
        str(peer_port),
        "-user",
        "100",
        "-password",
        sip_password,
        "-codec",
        codec,
        "-script",
        str(script_path),
        "-out",
        str(out_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=peer_log,
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("NRSP_")}
    env.update(
        {
            "NRSP_BRIDGE_CONTROL_ADDRESS": f"127.0.0.1:{control_port}",
            "NRSP_BRIDGE_AUTH_TOKEN": token,
            "NRSP_SIP_BIND_HOST": "127.0.0.1",
            "NRSP_SIP_PBX_HOST": "127.0.0.1",
            "NRSP_SIP_PBX_PORT": str(peer_port),
            "NRSP_SIP_USERNAME": "100",
            "NRSP_SIP_PASSWORD": sip_password,
            "NRSP_GATEWAY_ID": "sim-1",
            # Safe: the only reachable "PBX" is the loopback test peer above.
            "NRSP_LIVE_CALLS_ENABLED": "true",
            "NRSP_LIVE_CALL_ALLOWLIST": TEST_NUMBER,
            "NRSP_RTP_TIMEOUT_SECONDS": str(scenario.rtp_timeout_s),
        }
    )
    bridge = await asyncio.create_subprocess_exec(
        str(bridge_bin),
        env=env,
        stdout=bridge_log,
        stderr=subprocess.STDOUT,
    )
    result = RunResult(scenario.name, None, {}, out_dir)
    port = DiagoTelephonyPort(token, port=control_port)
    try:
        assert peer.stdout is not None
        await _wait_line(peer.stdout, "PEER_READY", 10)
        for _ in range(200):
            try:
                await port.connect()
                if await port.ready():
                    break
            except OSError:
                pass
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("bridge did not register with the test peer")
        started = time.perf_counter()
        result.session = await asyncio.wait_for(
            run_outbound_session(
                port,
                number=TEST_NUMBER,
                gateway_id="sim-1",
                call_id=f"e2e-{scenario.name}-{secrets.token_hex(3)}",
                voice_settings=settings,
                engine=build_engine(NRSP_PROFILE, TEST_CALL_DATA),
            ),
            scenario.max_call_s + 30,
        )
        result.timings["session_seconds"] = round(time.perf_counter() - started, 2)
        await asyncio.wait_for(peer.wait(), 20)
        result.peer = await asyncio.to_thread(_read_json, out_dir / "report.json")
        result.timings["dropped_caller_audio_frames"] = port.dropped_audio_frames
    except Exception as exc:  # recorded for the caller; tests assert on it
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        await port.close()
        for proc in (peer, bridge):
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        peer_log.close()
        bridge_log.close()
    await asyncio.to_thread(_write_summary, result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def run_scenarios(
    names: list[str], out_root: Path, settings: LocalVoicePipelineSettings
) -> list[RunResult]:
    binaries = build_go_binaries()
    t = time.perf_counter()
    preload_models(settings)
    preload = time.perf_counter() - t
    warm = warm_ollama(settings)
    results = []
    for name in names:
        cpu_before = cpu_performance_pct()
        result = await run_scenario(
            SCENARIOS[name], out_root / name, settings, binaries
        )
        result.timings.update(
            preload_models_s=round(preload, 2),
            warm_llm_s=round(warm, 2),
            cpu_perf_pct_before=cpu_before,
            cpu_perf_pct_after=cpu_performance_pct(),
        )
        results.append(result)
    return results


def main() -> None:
    import argparse

    from loguru import logger

    from nrsp_agent.voice.local_pipeline import load_local_voice_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="*", default=["happy_path"])
    parser.add_argument("--profile", default=str(ROOT / "config" / "demo.toml"))
    parser.add_argument("--out", default=str(ROOT / ".tmp" / "e2e"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(out / "pipeline.log", level="DEBUG", encoding="utf-8")
    logger.add(sys.stderr, level="WARNING")
    names = list(SCENARIOS) if args.scenarios == ["all"] else args.scenarios
    results = asyncio.run(
        run_scenarios(names, out, load_local_voice_settings(Path(args.profile)))
    )
    for result in results:
        print(digest(result))


def digest(result: RunResult) -> str:
    """Human-readable one-scenario report; full data is in summary.json."""
    s, peer = result.session, result.peer
    lines = [f"=== {result.scenario}  error={result.error}  timings={result.timings}"]
    if s is not None:
        lines.append(
            f"result={s.result} reason={s.reason} outcome={s.outcome} "
            f"result={s.call_result} guard_rejections={s.guard_rejections}"
        )
        lines += [
            f"  {t['at_ms']:>7} {t['role']:9} {t['source']:6} {t['text']}"
            for t in s.transcript
        ]
    for u in peer.get("utterances") or []:
        lines.append(
            f"  caller {u['name']:18} {u['start_ms']:>6}-{u['end_ms']:<6} "
            f"reply_latency_ms={u.get('reply_latency_ms')} "
            f"bot_stop_ms={u.get('bot_stop_ms')}"
        )
    if peer:
        rtp = peer.get("rtp") or {}
        lines.append(
            f"  peer result={peer.get('result')} hangup_by={peer.get('hangup_by')} "
            f"first_bot_speech_ms={peer.get('first_bot_speech_ms')} "
            f"bot_segments={len(peer.get('bot_segments') or [])}"
        )
        lines.append(
            f"  rtp packets={rtp.get('packets')} seq_gaps={rtp.get('seq_gaps')} "
            f"ts_errors={rtp.get('timestamp_errors')} ssrcs={rtp.get('ssrcs')} "
            f"jitter_ms={rtp.get('jitter_ms', 0):.2f} "
            f"drift_ms={rtp.get('clock_drift_ms', 0):.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

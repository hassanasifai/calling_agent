"""Opt-in end-to-end Urdu conversations over the real bridge, SIP and models.

Run with ``NRSP_E2E=1`` (needs Go, the local models and a running Ollama with
the configured model). No real telephone network is reachable: the bridge's
only PBX is the loopback SIP test peer.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from nrsp_agent.devtools.loopback import ROOT, SCENARIOS, RunResult, run_scenarios
from nrsp_agent.voice.local_pipeline import load_local_voice_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("NRSP_E2E") != "1", reason="set NRSP_E2E=1 to run real-model E2E"
)

SECRETS = ("پانچ ہزار", "پندرہ نومبر")


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunResult]:
    settings = load_local_voice_settings(ROOT / "config" / "demo.toml")
    out = Path(tmp_path_factory.mktemp("e2e"))
    runs = asyncio.run(run_scenarios(list(SCENARIOS), out, settings))
    return {r.scenario: r for r in runs}


def spoken(result: RunResult) -> str:
    assert result.session is not None
    return " ".join(
        t["text"] for t in result.session.transcript if t["role"] == "assistant"
    )


def assert_clean_media(result: RunResult) -> None:
    assert result.error is None, result.error
    rtp = result.peer["rtp"]
    assert rtp["ssrcs"] == 1 and rtp["seq_gaps"] == 0 and rtp["timestamp_errors"] == 0
    assert abs(rtp["clock_drift_ms"]) < 150
    peer = result.peer
    assert peer["register_authorized"] and peer["invite_authorized"]
    # One greeting, played promptly after answer.
    assert (
        peer["first_bot_speech_ms"] is not None and peer["first_bot_speech_ms"] < 2500
    )
    assert spoken(result).count("السلام علیکم") == 1


def test_happy_path(results: dict[str, RunResult]) -> None:
    r = results["happy_path"]
    assert_clean_media(r)
    assert r.session.outcome == "completed" and r.session.verified
    sources = [t["source"] for t in r.session.transcript if t["role"] == "assistant"]
    assert "llm" in sources
    assert r.peer["hangup_by"] == "bridge"
    assert all(u["reply_latency_ms"] is not None for u in r.peer["utterances"] or [])


def test_barge_in_stops_bot_quickly(results: dict[str, RunResult]) -> None:
    r = results["barge_in"]
    assert_clean_media(r)
    first = r.peer["utterances"][0]
    assert first["bot_speaking_at_start"]
    assert first["bot_stop_ms"] is not None and first["bot_stop_ms"] < 1000


@pytest.mark.parametrize(
    ("scenario", "outcome"),
    [
        ("opt_out", "opted_out"),
        ("wrong_person", "wrong_person"),
        ("silence", "no_response"),
    ],
)
def test_exits_never_disclose(
    results: dict[str, RunResult], scenario: str, outcome: str
) -> None:
    r = results[scenario]
    assert_clean_media(r)
    assert r.session.outcome == outcome and not r.session.verified
    assert not any(secret in spoken(r) for secret in SECRETS)
    assert r.peer["hangup_by"] == "bridge"


def test_remote_hangup_mid_turn_tears_down(results: dict[str, RunResult]) -> None:
    r = results["remote_hangup_mid_turn"]
    assert r.error is None, r.error
    assert r.session.result == "remote_hangup"
    assert r.peer["hangup_by"] == "peer"


def test_rtp_loss_ends_call(results: dict[str, RunResult]) -> None:
    r = results["rtp_loss"]
    assert r.error is None, r.error
    assert r.session.reason == "rtp_timeout"
    assert "rtp_lost" in [name for _, name in r.session.events]

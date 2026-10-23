"""Run ONE outbound session through an already running media bridge.

For the authorized real-phone test only. This tool places no call by itself:
the bridge dials only when it was started with ``NRSP_LIVE_CALLS_ENABLED=true``
and the number is in ``NRSP_LIVE_CALL_ALLOWLIST`` (see
docs/jazz-test-runbook.md). The bridge token is read from the environment,
never from the command line. Call data must be a test record, never a real
borrower.

    python -m nrsp_agent.devtools.live_call --number 9<authorized-number> \
        --profile-dir config/profiles/nrsp_verification \
        --call-data test_call.json --settings config/demo.toml --out .tmp/live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

from nrsp_agent.agent import build_engine
from nrsp_agent.session import run_outbound_session
from nrsp_agent.telephony.diago import DiagoTelephonyPort
from nrsp_agent.voice.local_pipeline import (
    load_local_voice_settings,
    preload_models,
)


async def run(args: argparse.Namespace) -> dict:
    token = os.environ.get("NRSP_BRIDGE_AUTH_TOKEN", "")
    host, _, port = os.environ.get(
        "NRSP_BRIDGE_CONTROL_ADDRESS", "127.0.0.1:8765"
    ).rpartition(":")
    settings = load_local_voice_settings(args.settings)
    call_data = json.loads(args.call_data.read_text(encoding="utf-8"))
    engine = build_engine(args.profile_dir, call_data)
    await asyncio.to_thread(preload_models, settings)
    async with DiagoTelephonyPort(token, host=host, port=int(port)) as bridge:
        if not await bridge.ready():
            raise SystemExit("bridge is not registered with the PBX")
        result = await run_outbound_session(
            bridge,
            number=args.number,
            gateway_id=args.gateway,
            call_id=f"live-{secrets.token_hex(4)}",
            voice_settings=settings,
            engine=engine,
        )
    return result.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", required=True, help="exact allowlisted string")
    parser.add_argument("--gateway", default="sim-1")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--call-data", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(".tmp/live"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from loguru import logger

    args.out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(args.out / "pipeline.log", level="DEBUG", encoding="utf-8")
    logger.add(sys.stderr, level="WARNING")
    summary = asyncio.run(run(args))
    path = args.out / f"{summary['call_id']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(f"result={summary['result']} outcome={summary['outcome']} -> {path}")


if __name__ == "__main__":
    main()

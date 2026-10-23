"""Per-turn latency breakdown from a loopback ``pipeline.log`` (Pipecat DEBUG).

For every caller turn: VAD stop -> transcript (ASR) -> LLM first token ->
LLM stream end -> TTS synthesis start -> "Bot started speaking". Times are
seconds after the VAD stop; the caller's real end of speech is ``stop_secs``
earlier (the peer report measures that full path as ``reply_latency_ms``).

    python scripts/turn_latency.py .tmp/e2e/pipeline.log [--since HH:MM:SS]
"""

from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})")
EVENTS = {
    "vad_stop": "VADUserStoppedSpeakingFrame received",
    "asr": "Transcription: [",
    "llm_ttfb": "EngineLLMService#",
    "tts_start": "Generating TTS [",
    "bot_audio": "Bot started speaking",
}


def parse(path: Path, since: str | None) -> list[dict[str, float]]:
    turns: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STAMP.match(line)
        if not m:
            continue
        if since and m.group(1)[11:19] < since:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        if EVENTS["vad_stop"] in line:
            current = {"vad_stop": t}
            turns.append(current)
        elif current is None:
            continue
        elif EVENTS["asr"] in line and "asr" not in current:
            current["asr"] = t
        elif (
            EVENTS["llm_ttfb"] in line and "TTFB:" in line and "llm_ttfb" not in current
        ):
            current["llm_ttfb"] = t
        elif EVENTS["llm_ttfb"] in line and "processing time" in line:
            current.setdefault("llm_end", t)
        elif EVENTS["tts_start"] in line and "tts_start" not in current:
            current["tts_start"] = t
        elif EVENTS["bot_audio"] in line and "bot_audio" not in current:
            current["bot_audio"] = t
    return [
        {k: round(v - turn["vad_stop"], 2) for k, v in turn.items() if k != "vad_stop"}
        for turn in turns
        if "asr" in turn and "bot_audio" in turn
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--since")
    args = parser.parse_args()
    rows = parse(args.log, args.since)
    keys = ["asr", "llm_ttfb", "llm_end", "tts_start", "bot_audio"]
    for row in rows:
        print("  ".join(f"{k}={row.get(k, '-')}" for k in keys))
    for k in keys:
        values = sorted(r[k] for r in rows if k in r)
        if values:
            p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
            p50 = statistics.median(values)
            print(f"{k:10s} n={len(values)} P50={p50:.2f} P95={p95:.2f}")


if __name__ == "__main__":
    main()

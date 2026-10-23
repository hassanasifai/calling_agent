"""End-to-end understanding: ASR transcript -> LLM action, per ASR model.

Uses the transcripts saved by ``benchmark_asr.py`` for the synthetic domain
set (voices other than the bot's own), runs each through the production
engine + LLM, and scores the chosen action. Ground-truth text is the upper
bound. Spoken-digit rows are scored with the code parser and the engine's
read-back rule only (the LLM never reconstructs numbers).

    .venv/bin/python scripts/benchmark_e2e_understanding.py \
        data/models/gguf/gemma-4-E2B_q4_0-it.gguf \
        data/eval/results/asr_canopy_canopy-m.json \
        data/eval/results/asr_fw_whisper-small-urdu_b5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_llm_turns import LLAMA, engine_at, start_server, stream  # noqa: E402

from nrsp_agent.agent.textnorm import extract_digits  # noqa: E402

# Domain-set intent labels (build_asr_eval.py) -> NRSP profile actions.
ACTIONS = {
    ("identity", "affirmative"): "confirm_identity",
    ("identity", "negative"): "wrong_person",
    ("identity", "who_is_calling"): "reply",
    ("discuss", "opt_out"): "opt_out",
    ("discuss", "handoff"): "human_handoff",
    ("discuss", "goodbye"): "finish",
    ("discuss", "ask_amount"): "tell_installment",
    ("discuss", "ask_date"): "tell_installment",
    ("discuss", "payment_difficulty"): "record_difficulty",
    ("discuss", "affirmative"): "record_promise",
    ("discuss", "unknown"): "record_promise",
}
VOICES = ("aegis", "mms")


def digits_ok(text: str, expected: str) -> bool:
    digits = extract_digits(text)
    if 4 < len(digits) <= 6:  # engine reads back the trailing four
        digits = digits[-4:]
    return digits == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("asr_results", type=Path, nargs="+")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/eval/asr/domain/manifest.json")
    )
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results"))
    args = parser.parse_args()
    manifest = {r["id"]: r for r in json.loads(args.manifest.read_text("utf-8"))}
    rows = [r for r in manifest.values() if r["voice"] in VOICES]
    sources = {"reference": {r["id"]: r["text"] for r in rows}}
    for path in args.asr_results:
        data = json.loads(path.read_text("utf-8"))
        sources[data["model"]] = {
            r["id"]: r["hyp"] for r in data["rows"] if r["set"] == "domain"
        }
    server = start_server(LLAMA, args.model, args.port, args.threads)
    report = {}
    try:
        for name, texts in sources.items():
            ok = n = dig_ok = dig_n = false_end = 0
            misses = []
            for row in rows:
                text = texts[row["id"]]
                if row["digits"]:
                    dig_n += 1
                    dig_ok += digits_ok(text, row["digits"])
                    continue
                expected = ACTIONS[(row["stage"], row["intent"])]
                engine = engine_at(row["stage"])
                system = {"role": "system", "content": engine.system_prompt()}
                schema = engine.response_schema()
                stream(args.port, [system, {"role": "user", "content": "."}], schema, 1)
                message = {"role": "user", "content": engine.turn_message(text, str)}
                raw, *_ = stream(args.port, [system, message], schema)
                try:
                    got = json.loads(raw)["action"]
                except (ValueError, KeyError):
                    got = "INVALID"
                n += 1
                ok += got == expected
                false_end += (
                    got in ("opt_out", "callback", "finish") and got != expected
                )
                if got != expected:
                    misses.append(f"{row['id']}: {text} -> {got} (want {expected})")
            report[name] = {
                "action_acc": f"{ok}/{n}",
                "digits_acc": f"{dig_ok}/{dig_n}",
                "false_unconfirmed_end": false_end,
                "misses": misses,
            }
            print(
                json.dumps(
                    {name: {k: v for k, v in report[name].items() if k != "misses"}}
                ),
                flush=True,
            )
    finally:
        server.terminate()
        server.wait(timeout=30)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"e2e_understanding_{args.model.stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    main()

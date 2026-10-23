"""Spoken-digit accuracy: Canopy-M CTC free decoding vs digit-grammar decoding.

The filler penalty is chosen on the bot's own voice (fasih) only and reported
on the other voices (aegis, mms), so the score is not tuned on its test data.

    .cache/venv-bench/bin/python scripts/benchmark_digit_grammar.py
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
from canopy.checkpoint import model_from_checkpoint
from canopy.inference.transcriber import Transcriber, _tokenizer_from_checkpoint
from ctc_digit_grammar import DigitGrammar

from nrsp_agent.agent.textnorm import extract_digits

EVAL = Path("data/eval/asr/domain")


def read(path: Path) -> torch.Tensor:
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


def ok(digits: str, expected: str) -> bool:
    if len(expected) < len(digits) <= len(expected) + 2:
        digits = digits[-len(expected) :]
    return digits == expected


def main() -> None:
    torch.set_num_threads(4)
    ck = torch.load("data/models/hf/Canopy-M/canopy-m.pt", weights_only=True)
    tokenizer = _tokenizer_from_checkpoint(ck)
    asr = Transcriber(model_from_checkpoint(ck), tokenizer)
    rows = [
        r
        for r in json.loads((EVAL / "manifest.json").read_text("utf-8"))
        if r["digits"]
    ]
    lps = {r["id"]: asr.log_probs(read(EVAL / r["wav"]), "urd").numpy() for r in rows}
    free = {r["id"]: asr.transcribe(read(EVAL / r["wav"]), "urd") for r in rows}

    def score(voices, penalty=None):
        sel = [r for r in rows if r["voice"] in voices]
        if penalty is None:
            return sum(
                ok(extract_digits(free[r["id"]]), r["digits"]) for r in sel
            ), len(sel)
        grammar = DigitGrammar(tokenizer.tokens, filler_penalty=penalty)
        return sum(ok(grammar.decode(lps[r["id"]]), r["digits"]) for r in sel), len(sel)

    tune = {p: score({"fasih"}, p)[0] for p in (0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 8.0)}
    best = max(tune, key=lambda p: (tune[p], p))
    grammar = DigitGrammar(tokenizer.tokens, filler_penalty=best)
    started = time.perf_counter()
    outputs = {r["id"]: grammar.decode(lps[r["id"]]) for r in rows}
    per_clip_ms = 1000 * (time.perf_counter() - started) / len(rows)
    report = {
        "tuning_on_fasih": tune,
        "chosen_penalty": best,
        "free_decoding_heldout": score({"aegis", "mms"}),
        "grammar_heldout": score({"aegis", "mms"}, best),
        "grammar_decode_ms_per_clip": round(per_clip_ms, 1),
        "outputs": {k: [free[k], outputs[k]] for k in outputs},
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    Path("data/eval/results/digit_grammar.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    main()

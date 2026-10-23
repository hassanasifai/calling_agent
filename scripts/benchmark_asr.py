"""Benchmark one ASR model on the telephone-band eval sets (bench venv).

One model per process, run sequentially; thread count is capped so the
shared VM keeps headroom. Whisper decoding mirrors the live
``SharedWhisperSTTService`` (beam 5, temperature 0, no conditioning, token
cap, stage-specific prompt, no-speech filter) unless overridden.

    .cache/venv-bench/bin/python scripts/benchmark_asr.py fw:data/models/ct2/X
    .cache/venv-bench/bin/python scripts/benchmark_asr.py canopy:<path/to/canopy-m.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import psutil
from jiwer import cer, wer

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import (
    detect_intent,
    digits_match,
    extract_digits,
    normalize,
)

TERMINAL = {"opt_out", "handoff", "goodbye"}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        pcm = wav.readframes(wav.getnframes())
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def load_whisper(path: str, threads: int, beam: int, prompts: bool):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        path,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        download_root="data/models/faster-whisper",
    )

    def run(audio: np.ndarray, stage: str | None, max_tokens: int) -> str:
        prompt = None
        if prompts and stage:
            prompt = (
                script.ASR_PROMPT_DIGITS
                if stage == "verify"
                else (script.ASR_PROMPT_DOMAIN)
            )
        segments, _ = model.transcribe(
            audio,
            language="ur",
            initial_prompt=prompt,
            beam_size=beam,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            max_new_tokens=max_tokens,
        )
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.4)

    return run


def load_canopy(path: str, threads: int):
    import torch
    from canopy.checkpoint import model_from_checkpoint
    from canopy.inference.transcriber import Transcriber, _tokenizer_from_checkpoint

    torch.set_num_threads(threads)
    # Safe unpickler: the upstream loader uses weights_only=False.
    ck = torch.load(path, map_location="cpu", weights_only=True)
    asr = Transcriber(model_from_checkpoint(ck), _tokenizer_from_checkpoint(ck))

    def run(audio: np.ndarray, stage: str | None, max_tokens: int) -> str:
        return asr.transcribe(audio, language="urd")

    return run


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 3) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="fw:<name-or-ct2-dir> or canopy:<pt>")
    parser.add_argument("--eval", type=Path, default=Path("data/eval/asr"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--no-prompts", action="store_true")
    parser.add_argument("--sets", default="noise,domain,fleurs")
    parser.add_argument("--out", type=Path, default=Path("data/eval/results"))
    args = parser.parse_args()
    kind, path = args.model.split(":", 1)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    started = time.perf_counter()
    if kind == "fw":
        run = load_whisper(path, args.threads, args.beam, not args.no_prompts)
    else:
        run = load_canopy(path, args.threads)
    load_s = time.perf_counter() - started
    cold_audio = read_wav(next((args.eval / "domain").glob("*.wav")))
    started = time.perf_counter()
    run(cold_audio, "identity", 96)
    first_s = time.perf_counter() - started

    label = f"{kind}:{Path(path).stem if '/' in path else path}"
    if kind == "fw":
        label += f":b{args.beam}" + (":noprompt" if args.no_prompts else "")
    report: dict = {
        "model": label,
        "threads": args.threads,
        "load_s": round(load_s, 2),
        "first_infer_s": round(first_s, 3),
        "rss_after_load_mb": round((proc.memory_info().rss - rss0) / 2**20),
    }
    cpu0 = proc.cpu_times()
    rows_out = []
    for set_name in args.sets.split(","):
        rows = json.loads((args.eval / set_name / "manifest.json").read_text("utf-8"))
        lat, audio_s, refs, hyps = [], 0.0, [], []
        intent_ok = intent_n = digit_ok = digit_n = false_terminal = 0
        halluc = []
        for row in rows:
            audio = read_wav(args.eval / set_name / row["wav"])
            t0 = time.perf_counter()
            hyp = run(audio, row.get("stage"), 224 if set_name == "fleurs" else 96)
            lat.append(time.perf_counter() - t0)
            audio_s += audio.size / 16_000
            rows_out.append(
                {"set": set_name, "id": row["id"], "ref": row["text"], "hyp": hyp}
            )
            if set_name == "noise":
                if normalize(hyp):
                    halluc.append(hyp)
                continue
            refs.append(normalize(row["text"]))
            hyps.append(normalize(hyp) or "∅")
            if row.get("digits"):
                digit_n += 1
                digit_ok += digits_match(extract_digits(hyp), row["digits"])
            elif row.get("intent"):
                got = detect_intent(hyp).value
                intent_n += 1
                intent_ok += got == row["intent"]
                false_terminal += got in TERMINAL and got != row["intent"]
        stats = {
            "n": len(rows),
            "lat_p50_s": percentile(lat, 50),
            "lat_p95_s": percentile(lat, 95),
            "lat_mean_s": round(float(np.mean(lat)), 3),
            "rtf": round(sum(lat) / audio_s, 3),
        }
        if set_name == "noise":
            stats["hallucinations"] = len(halluc)
            stats["hallucination_texts"] = halluc
        else:
            stats["wer"] = round(wer(refs, hyps), 4)
            stats["cer"] = round(cer(refs, hyps), 4)
        if set_name == "domain":
            stats["intent_acc"] = f"{intent_ok}/{intent_n}"
            stats["digit_acc"] = f"{digit_ok}/{digit_n}"
            stats["false_terminal_intents"] = false_terminal
        report[set_name] = stats
        print(json.dumps({set_name: stats}, ensure_ascii=False), flush=True)
    cpu1 = proc.cpu_times()
    report["cpu_s_total"] = round(cpu1.user + cpu1.system - cpu0.user - cpu0.system, 1)
    report["peak_rss_mb"] = round(proc.memory_info().rss / 2**20)
    args.out.mkdir(parents=True, exist_ok=True)
    slug = label.replace(":", "_").replace("/", "_")
    (args.out / f"asr_{slug}.json").write_text(
        json.dumps({**report, "rows": rows_out}, ensure_ascii=False, indent=1), "utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()

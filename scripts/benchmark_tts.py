"""Compare Urdu TTS voices: latency and intelligibility over G.711 (bench venv).

Intelligibility = character error rate of the most accurate measured ASR
(whisper-small-urdu) on the voice after an 8 kHz A-law round trip. It is a
proxy: naturalness needs human listening, so samples are written to
``data/eval/tts_samples/<voice>/`` (telephone band) for that.

    .cache/venv-bench/bin/python scripts/benchmark_tts.py
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import audioop
import numpy as np
import soxr
from faster_whisper import WhisperModel
from jiwer import cer
from piper import PiperVoice

from nrsp_agent.agent import build_engine
from nrsp_agent.agent.textnorm import normalize
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA

# Natural model-style replies (not from evaluation sets).
REPLIES = [
    "جی بالکل، میں آپ کی بات سمجھ گیا ہوں۔",
    "اس بارے میں تفصیل ہمارا نمائندہ آپ کو بتائے گا۔",
    "بہت شکریہ، ہم آپ کے تعاون کے شکر گزار ہیں۔",
    "میں آپ کی مشکل سمجھتا ہوں، آپ نے بتایا اس کا شکریہ۔",
]


def sentences() -> list[str]:
    engine = build_engine(NRSP_PROFILE, TEST_CALL_DATA)
    p = engine.profile
    texts = [
        engine.greeting().speak,
        engine.render(p.silence_prompt),
        engine.render(p.repeat_request),
    ]
    texts += [engine.render(f"{a.say} {a.ask}") for a in p.actions if a.say or a.ask]
    texts.append(p.readback.replace("{digits_spoken}", "چار، پانچ، چھ، سات"))
    return [t for t in texts if t] + REPLIES


def telephone(pcm: np.ndarray, rate: int) -> np.ndarray:
    x = soxr.resample(pcm.astype(np.float32) / 32768.0, rate, 8_000, quality="HQ")
    raw = np.clip(x * 32767, -32768, 32767).astype(np.int16).tobytes()
    y = np.frombuffer(audioop.alaw2lin(audioop.lin2alaw(raw, 2), 2), np.int16)
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voices", default="ur_PK-fasih-medium,ur_PK-aegis_female-medium"
    )
    parser.add_argument("--asr", default="data/models/ct2/whisper-small-urdu")
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    args = parser.parse_args()
    asr = WhisperModel(args.asr, device="cpu", compute_type="int8", cpu_threads=4)
    texts = sentences()
    report = {}
    for name in args.voices.split(","):
        voice = PiperVoice.load(f"data/models/piper/{name}.onnx")
        rate = voice.config.sample_rate
        list(voice.synthesize("گرم"))  # warm-up
        first, rtf, refs, hyps = [], [], [], []
        sample_dir = args.out / "tts_samples" / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            started = time.perf_counter()
            chunks = []
            for chunk in voice.synthesize(text):
                if not chunks:
                    first.append(time.perf_counter() - started)
                chunks.append(chunk.audio_int16_bytes)
            elapsed = time.perf_counter() - started
            pcm = np.frombuffer(b"".join(chunks), np.int16)
            rtf.append(elapsed / (pcm.size / rate))
            phone = telephone(pcm, rate)
            with wave.open(str(sample_dir / f"{i:02d}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8_000)
                wav.writeframes(phone.tobytes())
            audio16 = soxr.resample(phone.astype(np.float32) / 32768.0, 8_000, 16_000)
            segments, _ = asr.transcribe(
                audio16,
                language="ur",
                beam_size=5,
                temperature=0.0,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            refs.append(normalize(text))
            hyps.append(normalize(" ".join(s.text for s in segments)) or "∅")
        report[name] = {
            "n": len(texts),
            "first_audio_p50_s": round(float(np.percentile(first, 50)), 3),
            "first_audio_p95_s": round(float(np.percentile(first, 95)), 3),
            "rtf_mean": round(float(np.mean(rtf)), 3),
            "g711_asr_cer": round(cer(refs, hyps), 4),
            "worst": sorted(
                (
                    {"ref": r, "hyp": h, "cer": round(cer(r, h), 2)}
                    for r, h in zip(refs, hyps, strict=True)
                ),
                key=lambda x: -x["cer"],
            )[:3],
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "results" / "tts.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    main()

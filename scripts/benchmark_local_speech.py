"""Offline Urdu Piper-to-Whisper smoke test and latency measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soxr
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.download_voices import download_voice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("data/models"))
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--tts-voice", default="ur_PK-fasih-medium")
    parser.add_argument(
        "--text", default="السلام علیکم، یہ ایک مقامی اردو آواز کی آزمائش ہے۔"
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    piper_dir = args.model_root / "piper"
    piper_dir.mkdir(parents=True, exist_ok=True)
    voice_path = piper_dir / f"{args.tts_voice}.onnx"
    if not voice_path.exists():
        download_voice(args.tts_voice, piper_dir)

    load_tts_started = time.perf_counter()
    voice = PiperVoice.load(voice_path)
    tts_load_seconds = time.perf_counter() - load_tts_started

    tts_started = time.perf_counter()
    chunks = list(voice.synthesize(args.text))
    pcm = b"".join(chunk.audio_int16_bytes for chunk in chunks)
    tts_seconds = time.perf_counter() - tts_started
    samples = np.frombuffer(pcm, dtype=np.int16)
    audio_seconds = len(samples) / voice.config.sample_rate
    audio_16k = soxr.resample(
        samples.astype(np.float32) / 32768.0,
        voice.config.sample_rate,
        16_000,
        quality="HQ",
    )

    load_asr_started = time.perf_counter()
    asr = WhisperModel(
        args.asr_model,
        device="cpu",
        compute_type="int8",
        download_root=str(args.model_root / "faster-whisper"),
    )
    asr_load_seconds = time.perf_counter() - load_asr_started

    asr_started = time.perf_counter()
    segments, _ = asr.transcribe(audio_16k, language="ur", beam_size=1)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    asr_seconds = time.perf_counter() - asr_started

    print(
        json.dumps(
            {
                "source_text": args.text,
                "transcript": transcript,
                "audio_seconds": round(audio_seconds, 3),
                "tts_load_seconds": round(tts_load_seconds, 3),
                "tts_inference_seconds": round(tts_seconds, 3),
                "tts_realtime_factor": round(tts_seconds / audio_seconds, 3),
                "asr_load_seconds": round(asr_load_seconds, 3),
                "asr_inference_seconds": round(asr_seconds, 3),
                "asr_realtime_factor": round(asr_seconds / audio_seconds, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Synthesize telephone-quality Urdu caller utterances for offline tests.

These stand in for a human caller in the loopback SIP test. They are
synthetic (Piper, slower and with more prosody noise than the bot voice) and
are degraded to the telephone band with a little background noise. They prove
the pipeline end to end; they are NOT an accuracy benchmark, which needs real
recorded Urdu telephone speech.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import soxr
from piper import PiperVoice
from piper.config import SynthesisConfig

UTTERANCES: dict[str, str] = {
    "confirm_identity": "جی ہاں، میں احمد علی بات کر رہا ہوں۔",
    "barge_in_identity": "ہیلو، جی میں احمد علی بول رہا ہوں۔",
    "cnic_last4": "میرے شناختی کارڈ کے آخری چار ہندسے چار پانچ چھ سات ہیں۔",
    "cnic_wrong": "میرے آخری ہندسے ایک دو تین چار ہیں۔",
    "promise_to_pay": "جی ہاں، میں اگلے ہفتے تک قسط ادا کر دوں گا۔",
    "goodbye": "ٹھیک ہے، شکریہ۔ اللہ حافظ۔",
    "opt_out": "مجھے دوبارہ کال نہ کریں۔",
    "wrong_person": "نہیں، یہ غلط نمبر ہے۔",
    "handoff": "مجھے کسی نمائندے سے بات کرنی ہے۔",
    # Candidate caller replies for the post-verification (LLM) turn; the
    # loopback scenario uses whichever survives G.711 + Whisper-small best.
    "will_try": "ٹھیک ہے، میں کوشش کروں گا۔",
    "pay_on_time": "جی ہاں، میں وقت پر ادائیگی کر دوں گا۔",
    "pay_sure": "جی بالکل، میں ادائیگی کر دوں گا۔",
    "deposit_next_week": "جی، میں اگلے ہفتے قسط جمع کرا دوں گا۔",
    "payment_difficulty": "ابھی میرے پاس پیسے نہیں ہیں۔",
    # Answers to the engine's confirmation questions (digit read-back,
    # confirming an opt-out, handoff or wrong person).
    "yes_correct": "جی ہاں، یہ درست ہیں۔",
    "yes": "جی ہاں۔",
}


def synthesize(voice: PiperVoice, text: str, seed: int) -> np.ndarray:
    config = SynthesisConfig(length_scale=1.12, noise_scale=0.75, noise_w_scale=0.9)
    pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text, config))
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = soxr.resample(audio, voice.config.sample_rate, 8_000, quality="HQ")
    # Telephone band (300-3400 Hz) and a -40 dBFS noise floor.
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, d=1 / 8_000)
    spectrum[(freqs < 300) | (freqs > 3_400)] = 0
    audio = np.fft.irfft(spectrum, n=audio.size)
    rng = np.random.default_rng(seed)
    audio = 0.7 * audio / max(1e-6, float(np.max(np.abs(audio))))
    audio += rng.normal(0, 0.01, audio.size)
    lead = np.zeros(int(0.15 * 8_000), dtype=np.float32)
    audio = np.concatenate([lead, audio, lead])
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="data/models/piper/ur_PK-fasih-medium.onnx")
    parser.add_argument("--out", default="tests/fixtures/caller")
    parser.add_argument("--only", nargs="*", help="regenerate just these names")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(args.voice)
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.only and manifest_path.exists()
        else {}
    )
    for seed, (name, text) in enumerate(UTTERANCES.items()):
        if args.only and name not in args.only:
            continue
        samples = synthesize(voice, text, seed)
        with wave.open(str(out / f"{name}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8_000)
            wav.writeframes(samples.tobytes())
        manifest[name] = {"text": text, "seconds": round(samples.size / 8_000, 2)}
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

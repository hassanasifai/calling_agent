"""Build the telephone-band ASR evaluation sets (offline, bench venv).

Two sets, both passed through the same 8 kHz G.711 A-law channel the Jazz call
uses, then resampled to the 16 kHz the pipeline feeds ASR:

* ``fleurs``: human read Pakistani Urdu (Google FLEURS ``ur_pk`` test split,
  CC-BY-4.0). Real speakers, but read Wikipedia sentences, not call speech.
* ``domain``: call-specific utterances (CNIC digits, opt-out, handoff,
  goodbye, payment difficulty, Urdu-English code-switching) SYNTHESIZED with
  three different voices: Piper fasih (the bot's own voice), Piper
  aegis_female, and Meta MMS-TTS (a different model family; CC-BY-NC, used
  only for this internal evaluation). Synthetic speech is cleaner than real
  callers; results on it are an upper bound, not evidence of field accuracy.
* ``noise``: silence and line-noise clips for hallucination checks.

Run with the bench venv: ``.cache/venv-bench/bin/python scripts/build_asr_eval.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import audioop
import numpy as np
import soundfile as sf
import soxr

from nrsp_agent.conversation.policy import detect_intent, extract_digits

# (id, text, stage, expected intent, expected digits). Stage selects the ASR
# prompt exactly as the live policy does ("verify" uses the digit prompt).
DOMAIN: list[tuple[str, str, str, str, str]] = [
    ("id_yes", "جی ہاں، میں احمد علی بات کر رہا ہوں۔", "identity", "affirmative", ""),
    ("id_yes2", "جی میں ہی بول رہا ہوں۔", "identity", "affirmative", ""),
    ("id_yes_cs", "یس، میں احمد بول رہا ہوں۔", "identity", "affirmative", ""),
    ("id_wrong", "نہیں، یہ غلط نمبر ہے۔", "identity", "negative", ""),
    ("id_wrong2", "رانگ نمبر ہے جی۔", "identity", "negative", ""),
    ("id_who", "آپ کون بول رہے ہیں؟", "identity", "who_is_calling", ""),
    ("id_where", "آپ کہاں سے بات کر رہے ہیں؟", "identity", "who_is_calling", ""),
    ("d_plain", "چار پانچ چھ سات", "verify", "", "4567"),
    (
        "d_sentence",
        "میرے شناختی کارڈ کے آخری چار ہندسے چار پانچ چھ سات ہیں۔",
        "verify",
        "",
        "4567",
    ),
    ("d_zero", "ایک نو صفر آٹھ", "verify", "", "1908"),
    ("d_repeat", "تین تین سات دو", "verify", "", "3372"),
    ("d_lead", "جی، آخری ہندسے ہیں چھ صفر صفر ایک۔", "verify", "", "6001"),
    ("d_nine", "سات آٹھ نو دو", "verify", "", "7892"),
    ("d_english", "فور فائیو سکس سیون", "verify", "", "4567"),
    ("d_english2", "زیرو ون ٹو تھری", "verify", "", "0123"),
    ("o_call", "مجھے دوبارہ کال نہ کریں۔", "discuss", "opt_out", ""),
    ("o_phone", "براہ کرم آئندہ مجھے فون مت کریں۔", "discuss", "opt_out", ""),
    ("o_contact", "مجھ سے رابطہ نہ کریں، شکریہ۔", "discuss", "opt_out", ""),
    ("o_bother", "مجھے تنگ نہ کریں۔", "discuss", "opt_out", ""),
    ("o_cs", "پلیز مجھے کال مت کرو۔", "discuss", "opt_out", ""),
    ("o_list", "میرا نمبر اپنی لسٹ سے نکال دیں۔", "discuss", "opt_out", ""),
    ("h_rep", "مجھے کسی نمائندے سے بات کرنی ہے۔", "discuss", "handoff", ""),
    ("h_human", "کسی انسان سے بات کروائیں۔", "discuss", "handoff", ""),
    ("h_manager", "مینیجر سے بات کرائیں۔", "discuss", "handoff", ""),
    ("h_officer", "آپ مجھے کسی آفیسر سے ملا دیں۔", "discuss", "handoff", ""),
    ("h_person", "کیا میں کسی بندے سے بات کر سکتا ہوں؟", "discuss", "handoff", ""),
    ("g_allah", "ٹھیک ہے، شکریہ۔ اللہ حافظ۔", "discuss", "goodbye", ""),
    ("g_khuda", "اچھا جی، خدا حافظ۔", "discuss", "goodbye", ""),
    ("g_bye", "شکریہ، اوکے بائے۔", "discuss", "goodbye", ""),
    ("a_amount", "میری قسط کتنی ہے؟", "discuss", "ask_amount", ""),
    ("a_money", "مجھے کتنے پیسے دینے ہیں؟", "discuss", "ask_amount", ""),
    ("a_date", "آخری تاریخ کیا ہے؟", "discuss", "ask_date", ""),
    ("a_until", "یہ مجھے کب تک جمع کرانی ہے؟", "discuss", "ask_date", ""),
    ("p_money", "ابھی میرے پاس پیسے نہیں ہیں۔", "discuss", "payment_difficulty", ""),
    ("p_time", "مجھے تھوڑی مہلت چاہیے۔", "discuss", "payment_difficulty", ""),
    ("p_hard", "اس مہینے ادائیگی مشکل ہے۔", "discuss", "payment_difficulty", ""),
    (
        "p_job",
        "میری نوکری چلی گئی ہے، میں ابھی قسط نہیں دے سکتا۔",
        "discuss",
        "payment_difficulty",
        "",
    ),
    ("t_ontime", "جی ہاں، میں وقت پر ادائیگی کر دوں گا۔", "discuss", "affirmative", ""),
    ("t_week", "جی، میں اگلے ہفتے قسط جمع کرا دوں گا۔", "discuss", "affirmative", ""),
    ("t_try", "ٹھیک ہے، میں کوشش کروں گا۔", "discuss", "affirmative", ""),
    ("t_cs", "انشاءاللہ میں پیمنٹ کر دوں گا۔", "discuss", "unknown", ""),
    ("t_cs2", "اوکے، میں کل بینک جا کر جمع کروا دوں گا۔", "discuss", "affirmative", ""),
]


def telephone(audio: np.ndarray, rate: int, seed: int, noise: float = 0.004) -> bytes:
    """Band-limit to 8 kHz, add a light noise floor, G.711 A-law round trip,
    return 16 kHz PCM16 as the pipeline receives it."""
    x = soxr.resample(audio.astype(np.float32), rate, 8_000, quality="HQ")
    x = 0.6 * x / max(1e-6, float(np.max(np.abs(x))))
    x = x + np.random.default_rng(seed).normal(0, noise, x.size)
    pad = np.zeros(int(0.2 * 8_000), dtype=np.float32)
    x = np.concatenate([pad, x, pad])
    pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16).tobytes()
    decoded = audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2)
    y = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
    y = soxr.resample(y, 8_000, 16_000, quality="HQ")
    return np.clip(y * 32767, -32768, 32767).astype(np.int16).tobytes()


def write_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm16)


def piper_synth(model: Path, length_scale: float):
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    voice = PiperVoice.load(model)
    config = SynthesisConfig(
        length_scale=length_scale, noise_scale=0.7, noise_w_scale=0.9
    )

    def synth(text: str) -> tuple[np.ndarray, int]:
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text, config))
        return np.frombuffer(pcm, np.int16) / 32768.0, voice.config.sample_rate

    return synth


def mms_synth(model_dir: Path):
    import torch
    from transformers import AutoTokenizer, VitsModel

    torch.manual_seed(0)
    model = VitsModel.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def synth(text: str) -> tuple[np.ndarray, int]:
        with torch.no_grad():
            out = model(**tokenizer(text, return_tensors="pt")).waveform[0]
        return out.numpy(), model.config.sampling_rate

    return synth


def build_domain(out: Path, models: Path) -> list[dict]:
    voices = {
        "fasih": piper_synth(models / "piper/ur_PK-fasih-medium.onnx", 1.1),
        "aegis": piper_synth(models / "piper/ur_PK-aegis_female-medium.onnx", 1.05),
        "mms": mms_synth(models / "hf/mms-tts-urd"),
    }
    rows = []
    for v_index, (voice_name, synth) in enumerate(voices.items()):
        for u_index, (uid, text, stage, intent, digits) in enumerate(DOMAIN):
            audio, rate = synth(text)
            name = f"{uid}__{voice_name}"
            write_wav(
                out / f"{name}.wav", telephone(audio, rate, 1000 * v_index + u_index)
            )
            rows.append(
                {
                    "id": name,
                    "wav": f"{name}.wav",
                    "text": text,
                    "stage": stage,
                    "intent": intent,
                    "digits": digits,
                    "voice": voice_name,
                }
            )
    return rows


def build_fleurs(src: Path, out: Path, limit: int) -> list[dict]:
    rows = []
    with (src / "test.tsv").open(encoding="utf-8") as handle:
        records = sorted(csv.reader(handle, delimiter="\t"), key=lambda r: r[1])
    for index, record in enumerate(records[:limit]):
        audio, rate = sf.read(src / "test" / record[1], dtype="float32")
        name = Path(record[1]).stem
        write_wav(out / f"{name}.wav", telephone(audio, rate, index, noise=0.002))
        rows.append({"id": name, "wav": f"{name}.wav", "text": record[2]})
    return rows


def build_noise(out: Path) -> list[dict]:
    rng = np.random.default_rng(7)
    t = np.arange(int(3 * 16_000)) / 16_000
    clips = {
        "silence_3s": np.zeros(t.size),
        "silence_8s": np.zeros(8 * 16_000),
        "white_-50db": rng.normal(0, 0.003, t.size),
        "white_-35db": rng.normal(0, 0.018, t.size),
        "hum_50hz": 0.02 * np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.002, t.size),
        "breath": np.convolve(rng.normal(0, 0.05, t.size), np.ones(40) / 40, "same"),
        "clicks": np.where(rng.random(t.size) > 0.9995, 0.4, 0.0),
        "babble_low": sum(
            0.004 * np.sin(2 * np.pi * f * t + p)
            for f, p in zip(
                rng.uniform(150, 900, 12), rng.uniform(0, 6, 12), strict=True
            )
        )
        * (1 + np.sin(2 * np.pi * 3 * t)),
    }
    rows = []
    for index, (name, audio) in enumerate(clips.items()):
        pcm = telephone(np.asarray(audio, np.float32) + 1e-6, 16_000, index, noise=0.0)
        write_wav(out / f"{name}.wav", pcm)
        rows.append({"id": name, "wav": f"{name}.wav", "text": ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("data/models"))
    parser.add_argument("--fleurs", type=Path, default=Path("data/eval/fleurs_ur"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/asr"))
    parser.add_argument("--fleurs-limit", type=int, default=100)
    args = parser.parse_args()
    # Labels must agree with the live policy on the reference text; a mismatch
    # is a policy phrase gap, reported here rather than hidden in ASR scores.
    for uid, text, _, intent, digits in DOMAIN:
        got_intent, got_digits = detect_intent(text).value, extract_digits(text)
        if (intent and got_intent != intent) or (digits and digits not in got_digits):
            print(
                f"POLICY GAP {uid}: expected {intent or digits}, policy reads "
                f"{got_intent}/{got_digits}"
            )
    sets = {
        "domain": lambda d: build_domain(d, args.models),
        "fleurs": lambda d: build_fleurs(args.fleurs, d, args.fleurs_limit),
        "noise": build_noise,
    }
    for name, build in sets.items():
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = build(directory)
        (directory / "manifest.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{name}: {len(rows)} clips")


if __name__ == "__main__":
    main()

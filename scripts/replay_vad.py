"""Replay a caller recording through the production input path, offline.

Mirrors the live chain for one call leg: 8 kHz PCM -> G.711 A-law round trip
(the bridge decodes A-law) -> 20 ms frames -> Pipecat's SOXR stream resampler
to 16 kHz -> Silero VAD with the deployment's parameters. Prints every VAD
state change and, with ``--windows``, the per-window confidence and volume in
a time range, so a missed or late speech start can be explained from data.

    python scripts/replay_vad.py .tmp/e2e/opt_out/sent.wav \
        --profile config/ubuntu-dev.toml --windows 25.5-27.5
"""

from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

import audioop
import numpy as np
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

from nrsp_agent.voice.local_pipeline import load_local_voice_settings


def read_pcm8k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 8000 and wav.getsampwidth() == 2
        return wav.readframes(wav.getnframes())


class TracingSilero(SileroVADAnalyzer):
    """Silero analyzer that records every window it scores."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.windows: list[tuple[float, float]] = []

    def voice_confidence(self, buffer: bytes) -> float:
        confidence = float(np.asarray(super().voice_confidence(buffer)).ravel()[0])
        self.windows.append((confidence, 0.0))
        return confidence

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = super()._get_smoothed_volume(audio)
        conf, _ = self.windows[-1]
        self.windows[-1] = (conf, volume)
        return volume


async def replay(
    pcm8k: bytes, params: VADParams, reframe: bool = True
) -> tuple[list, list]:
    resampler = SOXRStreamAudioResampler(quality="HQ", clear_after_secs=None)
    vad = TracingSilero(sample_rate=16_000, params=params)
    vad.set_sample_rate(16_000)
    decoded = audioop.alaw2lin(audioop.lin2alaw(pcm8k, 2), 2)
    frame = 160 * 2
    # reframe=True mirrors DiagoInputTransport (20 ms frames); False replays
    # the resampler's raw ~100 ms bursts (the behaviour before the fix).
    out_frame = 320 * 2 if reframe else None
    reported, fed, changes, pending = VADState.QUIET, 0, [], b""
    for i in range(0, len(decoded) - frame + 1, frame):
        pending += await resampler.resample(decoded[i : i + frame], 8000, 16000)
        size = out_frame or len(pending)
        while pending and len(pending) >= size:
            chunk, pending = pending[:size], pending[size:]
            fed += len(chunk) // 2
            new = await vad.analyze_audio(chunk)
            # Pipecat's VADController reports only SPEAKING/QUIET changes.
            if new != reported and new in (VADState.SPEAKING, VADState.QUIET):
                changes.append((fed / 16000, new.name))
                reported = new
    await vad.cleanup()
    return changes, vad.windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("config/ubuntu-dev.toml"))
    parser.add_argument("--windows", help="start-end seconds to dump per window")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--start-secs", type=float)
    parser.add_argument("--min-volume", type=float)
    parser.add_argument("--bursts", action="store_true", help="pre-fix framing")
    args = parser.parse_args()
    s = load_local_voice_settings(args.profile)
    params = VADParams(
        confidence=args.confidence or s.vad_confidence,
        start_secs=args.start_secs or s.vad_start_secs,
        stop_secs=s.vad_stop_secs,
        **({"min_volume": args.min_volume} if args.min_volume is not None else {}),
    )
    print(params)
    changes, windows = asyncio.run(
        replay(read_pcm8k(args.wav), params, reframe=not args.bursts)
    )
    for at, name in changes:
        print(f"{at:8.2f}s  {name}")
    if args.windows:
        lo, hi = (float(x) for x in args.windows.split("-"))
        step = 512 / 16000
        for n, (conf, vol) in enumerate(windows):
            t = n * step
            if lo <= t <= hi:
                mark = (
                    "*"
                    if conf >= params.confidence and vol >= params.min_volume
                    else " "
                )
                print(f"{t:7.3f} conf={conf:.2f} vol={vol:.2f} {mark}")
    arr = np.array(windows)
    print(f"windows={len(arr)}")


if __name__ == "__main__":
    main()

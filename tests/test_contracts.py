import pytest
from pydantic import ValidationError

from nrsp_agent.telephony.contracts import AudioFrame


def test_audio_frame_rejects_partial_pcm_sample() -> None:
    with pytest.raises(ValidationError, match="whole 16-bit samples"):
        AudioFrame(
            call_id="call-1",
            pcm16=b"\x00\x00\x00",
            sample_rate=8_000,
            monotonic_ms=0,
        )


def test_audio_frame_is_immutable() -> None:
    frame = AudioFrame(
        call_id="call-1", pcm16=b"\x00\x00", sample_rate=8_000, monotonic_ms=0
    )
    with pytest.raises(ValidationError):
        frame.sample_rate = 16_000  # type: ignore[misc]

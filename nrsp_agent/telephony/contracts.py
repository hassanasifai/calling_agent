"""Transport-independent telephony contracts used by later production gates."""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CallEventType(StrEnum):
    DIALING = "DIALING"
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    HANGUP = "HANGUP"
    FAILED = "FAILED"
    RTP_READY = "RTP_READY"
    RTP_LOST = "RTP_LOST"


class CallHandle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1, max_length=128)
    gateway_id: str = Field(min_length=1, max_length=128)


class CallEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1, max_length=128)
    event: CallEventType
    monotonic_ms: int = Field(ge=0)
    reason: str | None = Field(default=None, max_length=256)
    sip_status: int | None = Field(default=None, ge=100, le=699)
    codec: str | None = Field(default=None, max_length=32)


class AudioFrame(BaseModel):
    """Little-endian signed PCM16 mono audio with monotonic capture time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1, max_length=128)
    pcm16: bytes = Field(min_length=2, max_length=1_920_000)
    sample_rate: int = Field(ge=8_000, le=48_000)
    channels: int = Field(default=1, ge=1, le=1)
    monotonic_ms: int = Field(ge=0)

    @field_validator("pcm16")
    @classmethod
    def pcm16_must_contain_whole_samples(cls, value: bytes) -> bytes:
        if len(value) % 2:
            raise ValueError("PCM16 data must contain whole 16-bit samples")
        return value


class TelephonyPort(Protocol):
    async def start_call(
        self, number: str, gateway_id: str, call_id: str
    ) -> CallHandle: ...

    async def hangup(self, call_id: str, reason: str, *, drain: bool = False) -> None:
        """End the call; with ``drain`` queued speech plays out before BYE."""
        ...

    def events(self, call_id: str) -> AsyncIterator[CallEvent]: ...

    def incoming_audio(self, call_id: str) -> AsyncIterator[AudioFrame]: ...

    async def send_audio(self, call_id: str, frame: AudioFrame) -> None: ...

    async def clear_audio(self, call_id: str) -> None:
        """Discard speech queued for playout, e.g. on caller barge-in."""
        ...

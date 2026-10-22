from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame

from nrsp_agent.telephony.contracts import AudioFrame, CallEvent, CallHandle
from nrsp_agent.voice.pipecat_transport import (
    DiagoPipecatTransport,
    DiagoTransportParams,
)


class FakeTelephonyPort:
    def __init__(self, incoming: list[AudioFrame] | None = None) -> None:
        self.incoming = incoming or []
        self.sent: list[AudioFrame] = []

    async def start_call(
        self, number: str, gateway_id: str, call_id: str
    ) -> CallHandle:
        return CallHandle(call_id=call_id, gateway_id=gateway_id)

    async def hangup(self, call_id: str, reason: str) -> None:
        return None

    async def events(self, call_id: str) -> AsyncIterator[CallEvent]:
        if False:
            yield

    async def incoming_audio(self, call_id: str) -> AsyncIterator[AudioFrame]:
        for frame in self.incoming:
            yield frame

    async def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        self.sent.append(frame)


def test_params_match_g711_media_contract() -> None:
    params = DiagoTransportParams()

    assert params.audio_in_sample_rate == 16_000
    assert params.audio_out_sample_rate == 8_000
    assert params.audio_out_10ms_chunks == 2
    assert params.audio_in_channels == params.audio_out_channels == 1


@pytest.mark.asyncio
async def test_input_converts_bridge_pcm_to_pipecat_frame() -> None:
    pcm = b"\x01\x02" * 160
    port = FakeTelephonyPort(
        [
            AudioFrame(
                call_id="call-1",
                pcm16=pcm,
                sample_rate=8_000,
                monotonic_ms=1,
            ),
            AudioFrame(
                call_id="call-1",
                pcm16=pcm,
                sample_rate=8_000,
                monotonic_ms=21,
            ),
        ]
    )
    transport = DiagoPipecatTransport(port, "call-1")
    input_transport = transport.input()
    input_transport.push_audio_frame = AsyncMock()

    await input_transport._read_bridge_audio()

    pipecat_frame = input_transport.push_audio_frame.await_args_list[0].args[0]
    assert isinstance(pipecat_frame, InputAudioRawFrame)
    assert pipecat_frame.audio
    assert len(pipecat_frame.audio) % 2 == 0
    assert pipecat_frame.sample_rate == 16_000
    assert pipecat_frame.num_channels == 1


@pytest.mark.asyncio
async def test_output_converts_pipecat_frame_to_bridge_pcm() -> None:
    pcm = b"\x03\x04" * 160
    port = FakeTelephonyPort()
    output_transport = DiagoPipecatTransport(port, "call-2").output()

    accepted = await output_transport.write_audio_frame(
        OutputAudioRawFrame(audio=pcm, sample_rate=8_000, num_channels=1)
    )

    assert accepted is True
    assert len(port.sent) == 1
    assert port.sent[0].call_id == "call-2"
    assert port.sent[0].pcm16 == pcm
    assert port.sent[0].sample_rate == 8_000

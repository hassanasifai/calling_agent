"""Pipecat transport backed by the Windows-native Diago media bridge."""

from __future__ import annotations

import asyncio
import time

from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from nrsp_agent.telephony.contracts import AudioFrame, TelephonyPort


class DiagoTransportParams(TransportParams):
    """Fixed media settings for the miniSIPServer G.711 call leg."""

    audio_in_enabled: bool = True
    audio_in_sample_rate: int | None = 16_000
    audio_in_channels: int = 1
    audio_out_enabled: bool = True
    audio_out_sample_rate: int | None = 8_000
    audio_out_channels: int = 1
    audio_out_10ms_chunks: int = 2
    audio_out_auto_silence: bool = False
    audio_out_end_silence_secs: int = 0


class DiagoInputTransport(BaseInputTransport):
    """Convert incoming bridge PCM into Pipecat input frames."""

    def __init__(
        self,
        port: TelephonyPort,
        call_id: str,
        params: DiagoTransportParams,
        **kwargs: object,
    ) -> None:
        super().__init__(params, **kwargs)
        self._port = port
        self._call_id = call_id
        self._target_sample_rate = params.audio_in_sample_rate or 16_000
        self._resampler = SOXRStreamAudioResampler(quality="HQ", clear_after_secs=None)
        self._bridge_reader_task: asyncio.Task[None] | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._bridge_reader_task = self.create_task(
            self._read_bridge_audio(), f"{self}::_read_bridge_audio"
        )

    async def stop(self, frame: EndFrame) -> None:
        await self._cancel_bridge_reader()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        await self._cancel_bridge_reader()
        await super().cancel(frame)

    async def cleanup(self) -> None:
        await self._cancel_bridge_reader()
        await super().cleanup()

    async def _read_bridge_audio(self) -> None:
        async for frame in self._port.incoming_audio(self._call_id):
            pcm16 = await self._resampler.resample(
                frame.pcm16, frame.sample_rate, self._target_sample_rate
            )
            if pcm16:
                await self._push_resampled(pcm16)
        tail = await self._resampler.flush()
        if tail:
            await self._push_resampled(tail)

    async def _push_resampled(self, pcm16: bytes) -> None:
        await self.push_audio_frame(
            InputAudioRawFrame(
                audio=pcm16,
                sample_rate=self._target_sample_rate,
                num_channels=1,
            )
        )

    async def _cancel_bridge_reader(self) -> None:
        if self._bridge_reader_task is not None:
            await self.cancel_task(self._bridge_reader_task)
            self._bridge_reader_task = None


class DiagoOutputTransport(BaseOutputTransport):
    """Send Pipecat output PCM to the bridge without handling RTP in Python."""

    def __init__(
        self,
        port: TelephonyPort,
        call_id: str,
        params: DiagoTransportParams,
        **kwargs: object,
    ) -> None:
        super().__init__(params, **kwargs)
        self._port = port
        self._call_id = call_id

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        await self._port.send_audio(
            self._call_id,
            AudioFrame(
                call_id=self._call_id,
                pcm16=frame.audio,
                sample_rate=frame.sample_rate,
                channels=frame.num_channels,
                monotonic_ms=time.monotonic_ns() // 1_000_000,
            ),
        )
        return True


class DiagoPipecatTransport(BaseTransport):
    """Expose one active Diago call as a Pipecat bidirectional transport."""

    def __init__(
        self,
        port: TelephonyPort,
        call_id: str,
        params: DiagoTransportParams | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._port = port
        self._call_id = call_id
        self._params = params or DiagoTransportParams()
        self._input: DiagoInputTransport | None = None
        self._output: DiagoOutputTransport | None = None

    def input(self) -> DiagoInputTransport:
        if self._input is None:
            self._input = DiagoInputTransport(
                self._port,
                self._call_id,
                self._params,
                name=self._input_name,
            )
        return self._input

    def output(self) -> DiagoOutputTransport:
        if self._output is None:
            self._output = DiagoOutputTransport(
                self._port,
                self._call_id,
                self._params,
                name=self._output_name,
            )
        return self._output

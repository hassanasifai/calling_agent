from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

import pytest

from nrsp_agent.telephony.contracts import AudioFrame, CallEventType
from nrsp_agent.telephony.diago import BridgeProtocolError, DiagoTelephonyPort


class FakeBridge:
    def __init__(self, token: str) -> None:
        self.token = token
        self.commands: list[dict[str, object]] = []
        self.server: asyncio.Server | None = None

    async def start(self) -> tuple[str, int]:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        address = self.server.sockets[0].getsockname()
        return str(address[0]), int(address[1])

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while line := await reader.readline():
                command = json.loads(line)
                self.commands.append(command)
                ok = command["auth_token"] == self.token
                response: dict[str, object] = {
                    "type": "response",
                    "request_id": command["request_id"],
                    "ok": ok,
                }
                if command["command"] == "health":
                    response["registered"] = True
                if not ok:
                    response["error"] = "unauthorized"
                writer.write(json.dumps(response).encode() + b"\n")
                if command["command"] == "start_call" and ok:
                    for event in ("DIALING", "ANSWERED", "RTP_READY"):
                        writer.write(
                            json.dumps(
                                {
                                    "type": "event",
                                    "call_id": command["call_id"],
                                    "event": event,
                                    "monotonic_ms": 10,
                                    **({"codec": "PCMA"} if event == "RTP_READY" else {}),
                                }
                            ).encode()
                            + b"\n"
                        )
                    writer.write(
                        json.dumps(
                            {
                                "type": "audio",
                                "call_id": command["call_id"],
                                "pcm_b64": base64.b64encode(b"\x01\x00" * 160).decode(),
                                "sample_rate": 8_000,
                                "channels": 1,
                                "encoding": "pcm_s16le",
                                "monotonic_ms": 20,
                            }
                        ).encode()
                        + b"\n"
                    )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.fixture
async def fake_bridge() -> AsyncIterator[tuple[FakeBridge, str, int]]:
    bridge = FakeBridge("test-token")
    host, port = await bridge.start()
    yield bridge, host, port
    await bridge.close()


async def test_health_and_call_events(
    fake_bridge: tuple[FakeBridge, str, int],
) -> None:
    _, host, port = fake_bridge
    async with DiagoTelephonyPort("test-token", host, port) as adapter:
        assert await adapter.ready()
        handle = await adapter.start_call("90000000000", "sim-1", "call-1")
        assert handle.call_id == "call-1"

        events = []
        iterator = adapter.events("call-1")
        for _ in range(3):
            events.append(await anext(iterator))
        assert [event.event for event in events] == [
            CallEventType.DIALING,
            CallEventType.ANSWERED,
            CallEventType.RTP_READY,
        ]
        assert events[-1].codec == "PCMA"

        frame = await anext(adapter.incoming_audio("call-1"))
        assert frame.pcm16 == b"\x01\x00" * 160
        assert frame.sample_rate == 8_000


async def test_send_audio_serializes_pcm(
    fake_bridge: tuple[FakeBridge, str, int],
) -> None:
    bridge, host, port = fake_bridge
    async with DiagoTelephonyPort("test-token", host, port) as adapter:
        frame = AudioFrame(
            call_id="call-1",
            pcm16=b"\x02\x00" * 160,
            sample_rate=8_000,
            monotonic_ms=0,
        )
        await adapter.send_audio("call-1", frame)
    command = bridge.commands[-1]
    assert base64.b64decode(str(command["pcm_b64"])) == frame.pcm16
    assert command["encoding"] == "pcm_s16le"


async def test_rejected_command_raises(
    fake_bridge: tuple[FakeBridge, str, int],
) -> None:
    _, host, port = fake_bridge
    async with DiagoTelephonyPort("wrong-token", host, port) as adapter:
        with pytest.raises(BridgeProtocolError, match="unauthorized"):
            await adapter.ready()


async def test_frame_call_id_must_match() -> None:
    adapter = DiagoTelephonyPort("test-token")
    frame = AudioFrame(
        call_id="call-a",
        pcm16=b"\x00\x00",
        sample_rate=8_000,
        monotonic_ms=0,
    )
    with pytest.raises(ValueError, match="does not match"):
        await adapter.send_audio("call-b", frame)


def test_control_host_must_be_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        DiagoTelephonyPort("test-token", "192.0.2.10")

"""Async Python adapter for the native Diago SIP/RTP sidecar."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import secrets
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nrsp_agent.telephony.contracts import (
    AudioFrame,
    CallEvent,
    CallEventType,
    CallHandle,
)


class BridgeProtocolError(RuntimeError):
    """The sidecar sent invalid data or rejected a command."""


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    request_id: str
    ok: bool
    error: str | None = None
    registered: bool | None = None


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    call_id: str
    event: CallEventType
    monotonic_ms: int = Field(ge=0)
    reason: str | None = None
    sip_status: int | None = None
    codec: str | None = None


class _Audio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    call_id: str
    pcm_b64: str
    sample_rate: int = Field(ge=8_000, le=48_000)
    channels: int = Field(ge=1, le=1)
    encoding: str
    monotonic_ms: int = Field(ge=0)


class DiagoTelephonyPort:
    """One authenticated connection to the local media bridge."""

    def __init__(
        self,
        auth_token: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        command_timeout: float = 10.0,
    ) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("the media bridge control connection must be loopback")
        if not auth_token:
            raise ValueError("auth_token must not be empty")
        self._auth_token = auth_token
        self._host = host
        self._port = port
        self._command_timeout = command_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[_Response]] = {}
        self._event_queues: dict[str, asyncio.Queue[CallEvent | None]] = {}
        self._audio_queues: dict[str, asyncio.Queue[AudioFrame | None]] = {}
        self._write_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_monotonic_ms: dict[str, int] = {}

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._writer is not None:
                return
            self._reader, self._writer = await asyncio.open_connection(
                self._host, self._port
            )
            self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        task, self._reader_task = self._reader_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._reader = None
        self._fail_pending(BridgeProtocolError("media bridge connection closed"))

    async def __aenter__(self) -> DiagoTelephonyPort:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def ready(self) -> bool:
        response = await self._command("health")
        return response.registered is True

    async def start_call(
        self, number: str, gateway_id: str, call_id: str
    ) -> CallHandle:
        self._event_queues.setdefault(call_id, asyncio.Queue(maxsize=256))
        self._audio_queues.setdefault(call_id, asyncio.Queue(maxsize=512))
        try:
            await self._command(
                "start_call",
                call_id=call_id,
                gateway_id=gateway_id,
                number=number,
            )
        except Exception:
            self._event_queues.pop(call_id, None)
            self._audio_queues.pop(call_id, None)
            raise
        return CallHandle(call_id=call_id, gateway_id=gateway_id)

    async def hangup(self, call_id: str, reason: str) -> None:
        await self._command("hangup", call_id=call_id, reason=reason)

    async def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        if frame.call_id != call_id:
            raise ValueError("audio frame call_id does not match target call_id")
        await self._command(
            "send_audio",
            call_id=call_id,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            encoding="pcm_s16le",
            pcm_b64=base64.b64encode(frame.pcm16).decode("ascii"),
        )

    async def events(self, call_id: str) -> AsyncIterator[CallEvent]:
        queue = self._event_queues.setdefault(call_id, asyncio.Queue(maxsize=256))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
                if event.event in {CallEventType.HANGUP, CallEventType.FAILED}:
                    return
        finally:
            if self._event_queues.get(call_id) is queue:
                self._event_queues.pop(call_id, None)

    async def incoming_audio(self, call_id: str) -> AsyncIterator[AudioFrame]:
        queue = self._audio_queues.setdefault(call_id, asyncio.Queue(maxsize=512))
        try:
            while True:
                frame = await queue.get()
                if frame is None:
                    return
                yield frame
        finally:
            if self._audio_queues.get(call_id) is queue:
                self._audio_queues.pop(call_id, None)

    async def _command(self, command: str, **payload: object) -> _Response:
        await self.connect()
        if self._writer is None:
            raise BridgeProtocolError("media bridge is not connected")
        request_id = secrets.token_hex(12)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_Response] = loop.create_future()
        self._pending[request_id] = future
        message = {
            "type": "command",
            "request_id": request_id,
            "auth_token": self._auth_token,
            "command": command,
            **payload,
        }
        try:
            encoded = json.dumps(message, separators=(",", ":")).encode() + b"\n"
            async with self._write_lock:
                self._writer.write(encoded)
                await self._writer.drain()
            response = await asyncio.wait_for(future, self._command_timeout)
        finally:
            self._pending.pop(request_id, None)
        if not response.ok:
            raise BridgeProtocolError(response.error or "media bridge rejected command")
        return response

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while line := await self._reader.readline():
                if len(line) > 2_600_000:
                    raise BridgeProtocolError(
                        "media bridge message exceeded size limit"
                    )
                self._route_message(json.loads(line))
            raise BridgeProtocolError("media bridge disconnected")
        except asyncio.CancelledError:
            raise
        except (
            OSError,
            asyncio.QueueFull,
            binascii.Error,
            json.JSONDecodeError,
            ValidationError,
            BridgeProtocolError,
        ) as exc:
            error = BridgeProtocolError(f"media bridge read failed: {exc}")
            self._fail_pending(error)
            self._close_streams()
        finally:
            writer, self._writer = self._writer, None
            self._reader = None
            if writer is not None:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()

    def _route_message(self, raw: dict[str, Any]) -> None:
        message_type = raw.get("type")
        if message_type == "response":
            response = _Response.model_validate(raw)
            future = self._pending.get(response.request_id)
            if future is not None and not future.done():
                future.set_result(response)
            return
        if message_type == "event":
            event = _Event.model_validate(raw)
            public_event = CallEvent.model_validate(event.model_dump(exclude={"type"}))
            self._last_monotonic_ms[event.call_id] = event.monotonic_ms
            queue = self._event_queues.get(event.call_id)
            if queue is not None:
                queue.put_nowait(public_event)
            if event.event in {CallEventType.HANGUP, CallEventType.FAILED}:
                audio_queue = self._audio_queues.get(event.call_id)
                if audio_queue is not None:
                    audio_queue.put_nowait(None)
            return
        if message_type == "audio":
            audio = _Audio.model_validate(raw)
            if audio.encoding != "pcm_s16le":
                raise BridgeProtocolError("unsupported bridge audio encoding")
            pcm = base64.b64decode(audio.pcm_b64, validate=True)
            frame = AudioFrame(
                call_id=audio.call_id,
                pcm16=pcm,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                monotonic_ms=audio.monotonic_ms,
            )
            self._last_monotonic_ms[audio.call_id] = audio.monotonic_ms
            queue = self._audio_queues.get(audio.call_id)
            if queue is not None:
                queue.put_nowait(frame)
            return
        raise BridgeProtocolError("unknown media bridge message type")

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    def _close_streams(self) -> None:
        for call_id, queue in self._event_queues.items():
            failed = CallEvent(
                call_id=call_id,
                event=CallEventType.FAILED,
                monotonic_ms=self._last_monotonic_ms.get(call_id, 0),
                reason="bridge_disconnected",
            )
            self._force_put(queue, failed)
        for queue in self._audio_queues.values():
            self._force_put(queue, None)

    @staticmethod
    def _force_put(
        queue: asyncio.Queue[CallEvent | AudioFrame | None],
        value: CallEvent | AudioFrame | None,
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(value)

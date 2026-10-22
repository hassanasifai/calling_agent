from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import nrsp_agent.session as session_module
from nrsp_agent.session import SessionResultType, run_outbound_session
from nrsp_agent.telephony.contracts import (
    AudioFrame,
    CallEvent,
    CallEventType,
    CallHandle,
)
from nrsp_agent.voice.local_pipeline import LocalVoicePipelineSettings


class FailedCallPort:
    def __init__(self) -> None:
        self.started: tuple[str, str, str] | None = None

    async def start_call(
        self, number: str, gateway_id: str, call_id: str
    ) -> CallHandle:
        self.started = (number, gateway_id, call_id)
        return CallHandle(call_id=call_id, gateway_id=gateway_id)

    async def hangup(self, call_id: str, reason: str) -> None:
        return None

    async def events(self, call_id: str) -> AsyncIterator[CallEvent]:
        yield CallEvent(
            call_id=call_id,
            event=CallEventType.FAILED,
            monotonic_ms=10,
            reason="busy",
            sip_status=486,
        )

    async def incoming_audio(self, call_id: str) -> AsyncIterator[AudioFrame]:
        if False:
            yield

    async def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        return None


class AnsweredCallPort(FailedCallPort):
    async def events(self, call_id: str) -> AsyncIterator[CallEvent]:
        yield CallEvent(
            call_id=call_id,
            event=CallEventType.ANSWERED,
            monotonic_ms=10,
            sip_status=200,
        )
        await asyncio.sleep(0)
        yield CallEvent(
            call_id=call_id,
            event=CallEventType.HANGUP,
            monotonic_ms=20,
            reason="remote_bye",
        )


class FakeWorker:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def event_handler(self, name: str):
        def register(handler):
            self.handlers[name] = handler
            return handler

        return register


class FakeAgent:
    def __init__(self) -> None:
        self.worker = FakeWorker()
        self.greetings = 0

    async def queue_initial_greeting(self) -> None:
        self.greetings += 1


class FakeRunner:
    def __init__(self, *, handle_sigint: bool) -> None:
        assert handle_sigint is False
        self.worker: FakeWorker | None = None
        self.cancelled = asyncio.Event()

    async def add_workers(self, worker: FakeWorker) -> None:
        self.worker = worker

    async def run(self) -> None:
        assert self.worker is not None
        handler = self.worker.handlers["on_pipeline_started"]
        await handler(self.worker, object())
        await self.cancelled.wait()

    async def cancel(self, reason: str | None = None) -> None:
        self.cancelled.set()


@pytest.mark.asyncio
async def test_failed_call_never_constructs_voice_pipeline() -> None:
    port = FailedCallPort()

    result = await run_outbound_session(
        port,
        number="test-number",
        gateway_id="sim-1",
        call_id="call-1",
        voice_settings=LocalVoicePipelineSettings(),
    )

    assert port.started == ("test-number", "sim-1", "call-1")
    assert result.result == SessionResultType.CALL_FAILED
    assert result.reason == "busy"
    assert result.sip_status == 486


@pytest.mark.asyncio
async def test_remote_hangup_stops_answered_pipeline(monkeypatch) -> None:
    port = AnsweredCallPort()
    agent = FakeAgent()
    monkeypatch.setattr(session_module, "build_local_voice_agent", lambda *_: agent)
    monkeypatch.setattr(session_module, "WorkerRunner", FakeRunner)

    result = await run_outbound_session(
        port,
        number="test-number",
        gateway_id="sim-1",
        call_id="call-2",
        voice_settings=LocalVoicePipelineSettings(),
    )

    assert result.result == SessionResultType.REMOTE_HANGUP
    assert result.reason == "remote_bye"
    assert agent.greetings == 1

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import nrsp_agent.session as session_module
from nrsp_agent.agent import build_engine
from nrsp_agent.conversation.processors import ConversationRecord
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA
from nrsp_agent.session import SessionResultType, run_outbound_session
from nrsp_agent.telephony.contracts import (
    AudioFrame,
    CallEvent,
    CallEventType,
    CallHandle,
)
from nrsp_agent.voice.local_pipeline import LocalVoicePipelineSettings


def event(call_id: str, kind: CallEventType, **extra: object) -> CallEvent:
    return CallEvent(call_id=call_id, event=kind, monotonic_ms=10, **extra)


class ScriptedPort:
    """Telephony port whose event stream is driven by the test."""

    def __init__(self) -> None:
        self.started: tuple[str, str, str] | None = None
        self.queue: asyncio.Queue[CallEvent] = asyncio.Queue()
        self.hangups: list[tuple[str, bool]] = []
        self.hangup_error: Exception | None = None

    async def start_call(
        self, number: str, gateway_id: str, call_id: str
    ) -> CallHandle:
        self.started = (number, gateway_id, call_id)
        return CallHandle(call_id=call_id, gateway_id=gateway_id)

    async def hangup(self, call_id: str, reason: str, *, drain: bool = False) -> None:
        self.hangups.append((reason, drain))
        if self.hangup_error:
            raise self.hangup_error
        await self.queue.put(event(call_id, CallEventType.HANGUP, reason=reason))

    async def events(self, call_id: str) -> AsyncIterator[CallEvent]:
        while True:
            item = await self.queue.get()
            yield item
            if item.event in {CallEventType.HANGUP, CallEventType.FAILED}:
                return

    async def incoming_audio(self, call_id: str) -> AsyncIterator[AudioFrame]:
        if False:
            yield

    async def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        return None

    async def clear_audio(self, call_id: str) -> None:
        return None


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
        self.record = ConversationRecord()
        self.greetings = 0
        self.greeted = asyncio.Event()

    async def queue_initial_greeting(self) -> None:
        self.greetings += 1
        self.greeted.set()

    async def prewarm_llm(self, base_url: str, model: str) -> None:
        self.prewarmed = (base_url, model)


class FakeRunner:
    """Fires on_pipeline_started twice to prove the greeting is idempotent."""

    def __init__(self, *, handle_sigint: bool) -> None:
        assert handle_sigint is False
        self.worker: FakeWorker | None = None
        self.cancelled = asyncio.Event()
        self.ends_by_itself = FakeRunner.ends_by_itself_default

    ends_by_itself_default = False

    async def add_workers(self, worker: FakeWorker) -> None:
        self.worker = worker

    async def run(self) -> None:
        assert self.worker is not None
        handler = self.worker.handlers["on_pipeline_started"]
        await handler(self.worker, object())
        await handler(self.worker, object())
        if self.ends_by_itself:
            return
        await self.cancelled.wait()

    async def cancel(self, reason: str | None = None) -> None:
        self.cancelled.set()


@pytest.fixture
def agent(monkeypatch) -> FakeAgent:
    fake = FakeAgent()
    built_before_dial: list[bool] = []

    def build(*_: object) -> FakeAgent:
        built_before_dial.append(True)
        return fake

    monkeypatch.setattr(session_module, "build_local_voice_agent", build)
    monkeypatch.setattr(session_module, "WorkerRunner", FakeRunner)
    FakeRunner.ends_by_itself_default = False
    return fake


async def run(port: ScriptedPort, call_id: str = "call-1"):
    return await asyncio.wait_for(
        run_outbound_session(
            port,
            number="test-number",
            gateway_id="sim-1",
            call_id=call_id,
            voice_settings=LocalVoicePipelineSettings(),
            engine=build_engine(NRSP_PROFILE, TEST_CALL_DATA),
        ),
        5,
    )


async def test_failed_call_never_starts_pipeline(agent: FakeAgent) -> None:
    port = ScriptedPort()
    await port.queue.put(event("call-1", CallEventType.DIALING))
    await port.queue.put(
        event("call-1", CallEventType.FAILED, reason="sip_486", sip_status=486)
    )

    result = await run(port)

    assert port.started == ("test-number", "sim-1", "call-1")
    assert result.result == SessionResultType.CALL_FAILED
    assert result.reason == "sip_486" and result.sip_status == 486
    assert agent.greetings == 0


async def test_hangup_between_answer_and_rtp_ready_never_greets(
    agent: FakeAgent,
) -> None:
    port = ScriptedPort()
    await port.queue.put(event("call-1", CallEventType.ANSWERED, sip_status=200))
    await port.queue.put(
        event("call-1", CallEventType.HANGUP, reason="media_setup_failed")
    )

    result = await run(port)

    assert result.result == SessionResultType.REMOTE_HANGUP
    assert result.reason == "media_setup_failed"
    assert agent.greetings == 0


async def test_greets_once_after_rtp_ready_then_remote_hangup(
    agent: FakeAgent,
) -> None:
    port = ScriptedPort()
    for kind in (
        CallEventType.DIALING,
        CallEventType.ANSWERED,
        CallEventType.RTP_READY,
    ):
        await port.queue.put(event("call-1", kind))

    async def remote_bye() -> None:
        await agent.greeted.wait()
        await port.queue.put(
            event("call-1", CallEventType.HANGUP, reason="remote_hangup")
        )

    asyncio.create_task(remote_bye())
    result = await run(port)

    assert agent.greetings == 1
    assert result.result == SessionResultType.REMOTE_HANGUP
    assert port.hangups == []
    names = [name for _, name in result.events]
    assert (
        names.index("answered")
        < names.index("rtp_ready")
        < names.index("greeting_queued")
    )


async def test_pipeline_end_hangs_up_with_drain(agent: FakeAgent) -> None:
    FakeRunner.ends_by_itself_default = True
    port = ScriptedPort()
    await port.queue.put(event("call-1", CallEventType.RTP_READY))

    result = await run(port)

    assert result.result == SessionResultType.PIPELINE_FINISHED
    assert port.hangups == [("in_progress", True)]
    assert "hangup" in [name for _, name in result.events]


async def test_local_hangup_racing_remote_bye_is_not_an_error(
    agent: FakeAgent,
) -> None:
    FakeRunner.ends_by_itself_default = True
    port = ScriptedPort()
    port.hangup_error = RuntimeError("call_not_found")
    await port.queue.put(event("call-1", CallEventType.RTP_READY))
    await port.queue.put(event("call-1", CallEventType.HANGUP, reason="remote_hangup"))

    result = await run(port)

    assert result.result in {
        SessionResultType.PIPELINE_FINISHED,
        SessionResultType.REMOTE_HANGUP,
    }

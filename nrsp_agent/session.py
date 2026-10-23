"""Lifecycle coordination between one SIP call and one Pipecat worker."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from enum import StrEnum

from loguru import logger
from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel, ConfigDict, Field

from nrsp_agent.agent.engine import CallEngine
from nrsp_agent.conversation.processors import ConversationRecord
from nrsp_agent.telephony.contracts import CallEvent, CallEventType, TelephonyPort
from nrsp_agent.voice.local_pipeline import (
    LocalVoicePipelineSettings,
    build_local_voice_agent,
)
from nrsp_agent.voice.pipecat_transport import DiagoPipecatTransport

_TERMINAL = {CallEventType.HANGUP, CallEventType.FAILED}


class SessionResultType(StrEnum):
    REMOTE_HANGUP = "remote_hangup"
    CALL_FAILED = "call_failed"
    PIPELINE_FINISHED = "pipeline_finished"


class SessionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    result: SessionResultType
    reason: str | None = None
    sip_status: int | None = None
    outcome: str | None = None
    # Profile-defined structured result: flags reached and fields recorded.
    call_result: dict[str, object] = Field(default_factory=dict)
    transcript: list[dict[str, object]] = Field(default_factory=list)
    events: list[tuple[int, str]] = Field(default_factory=list)
    metrics: list[dict[str, object]] = Field(default_factory=list)
    guard_rejections: int = 0


async def run_outbound_session(
    port: TelephonyPort,
    *,
    number: str,
    gateway_id: str,
    call_id: str,
    voice_settings: LocalVoicePipelineSettings,
    engine: CallEngine,
) -> SessionResult:
    """Run one allowlisted outbound call until SIP or the AI pipeline ends."""
    # Build before dialing: model loading must not delay the greeting.
    agent = build_local_voice_agent(
        DiagoPipecatTransport(port, call_id), voice_settings, engine
    )
    record = agent.record

    def result(kind: SessionResultType, event: CallEvent | None, reason: str | None):
        return SessionResult(
            call_id=call_id,
            result=kind,
            reason=reason,
            sip_status=event.sip_status if event else None,
            outcome=engine.outcome,
            call_result=engine.result(),
            transcript=[asdict(turn) for turn in record.turns],
            events=list(record.events),
            metrics=list(record.metrics),
            guard_rejections=len(record.guard_rejections),
        )

    prewarm = asyncio.create_task(
        agent.prewarm_llm(
            voice_settings.llm_base_url,
            voice_settings.llm_model,
            voice_settings.auth_headers(),
        )
    )
    await port.start_call(number, gateway_id, call_id)
    events = port.events(call_id)
    media_event = await _wait_for_media(events, record)
    if media_event.event in _TERMINAL:
        kind = (
            SessionResultType.CALL_FAILED
            if media_event.event == CallEventType.FAILED
            else SessionResultType.REMOTE_HANGUP
        )
        await agent.aclose()
        return result(kind, media_event, media_event.reason)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(agent.worker)
    greeted = False

    @agent.worker.event_handler("on_pipeline_started")
    async def queue_greeting(*_: object) -> None:
        nonlocal greeted
        if not greeted:
            greeted = True
            record.add_event("greeting_queued")
            await agent.queue_initial_greeting()

    runner_task = asyncio.create_task(runner.run())
    terminal_task = asyncio.create_task(
        _wait_for_terminal(events, record, agent.on_keypad)
    )
    try:
        done, _ = await asyncio.wait(
            {runner_task, terminal_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if terminal_task in done:
            event = terminal_task.result()
            await runner.cancel(reason=event.reason or event.event.value)
            await runner_task
            kind = (
                SessionResultType.CALL_FAILED
                if event.event == CallEventType.FAILED
                else SessionResultType.REMOTE_HANGUP
            )
            return result(kind, event, event.reason)

        await runner_task
        reason = engine.outcome
        record.add_event("local_hangup")
        await _hangup_quietly(port, call_id, reason, drain=True)
        # The bridge reports HANGUP once the drained closing speech has played.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(terminal_task), 15)
        return result(SessionResultType.PIPELINE_FINISHED, None, reason)
    except asyncio.CancelledError:
        await _hangup_quietly(port, call_id, "session_cancelled", drain=False)
        await runner.cancel(reason="session_cancelled")
        raise
    finally:
        if not prewarm.done():
            prewarm.cancel()
        await agent.aclose()
        if not terminal_task.done():
            terminal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await terminal_task


async def _hangup_quietly(
    port: TelephonyPort, call_id: str, reason: str, *, drain: bool
) -> None:
    try:
        await port.hangup(call_id, reason, drain=drain)
    except Exception as exc:  # the call may already have ended remotely
        logger.debug(f"hangup for {call_id} not applied: {exc}")


async def _wait_for_media(
    events: AsyncIterator[CallEvent], record: ConversationRecord
) -> CallEvent:
    """Wait until RTP is ready, so nothing is sent before media can flow."""
    async for event in events:
        record.add_event(event.event.value.lower())
        if event.event == CallEventType.RTP_READY or event.event in _TERMINAL:
            return event
    raise RuntimeError("call event stream ended before media was ready")


async def _wait_for_terminal(
    events: AsyncIterator[CallEvent],
    record: ConversationRecord,
    on_keypad: Callable[[str], Awaitable[None]],
) -> CallEvent:
    async for event in events:
        if event.event == CallEventType.DTMF:
            if event.digit:
                await on_keypad(event.digit)
            continue
        record.add_event(event.event.value.lower())
        if event.event in _TERMINAL:
            return event
    raise RuntimeError("call event stream ended without terminal event")

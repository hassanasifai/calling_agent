"""Lifecycle coordination between one SIP call and one Pipecat worker."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from enum import StrEnum

from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel, ConfigDict

from nrsp_agent.telephony.contracts import CallEvent, CallEventType, TelephonyPort
from nrsp_agent.voice.local_pipeline import (
    LocalVoicePipelineSettings,
    build_local_voice_agent,
)
from nrsp_agent.voice.pipecat_transport import DiagoPipecatTransport


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


async def run_outbound_session(
    port: TelephonyPort,
    *,
    number: str,
    gateway_id: str,
    call_id: str,
    voice_settings: LocalVoicePipelineSettings,
) -> SessionResult:
    """Run one allowlisted outbound call until SIP or the AI pipeline ends."""
    await port.start_call(number, gateway_id, call_id)
    events = port.events(call_id)
    answered = await _wait_for_answer(events)
    if answered.event == CallEventType.FAILED:
        return _terminal_result(answered)

    transport = DiagoPipecatTransport(port, call_id)
    agent = build_local_voice_agent(transport, voice_settings)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(agent.worker)

    @agent.worker.event_handler("on_pipeline_started")
    async def queue_greeting(*_: object) -> None:
        await agent.queue_initial_greeting()

    runner_task = asyncio.create_task(runner.run())
    terminal_task = asyncio.create_task(_wait_for_terminal(events))
    try:
        done, _ = await asyncio.wait(
            {runner_task, terminal_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if terminal_task in done:
            event = terminal_task.result()
            await runner.cancel(reason=event.reason or event.event.value)
            await runner_task
            return _terminal_result(event)

        await runner_task
        await port.hangup(call_id, "pipeline_finished")
        return SessionResult(
            call_id=call_id,
            result=SessionResultType.PIPELINE_FINISHED,
            reason="pipeline_finished",
        )
    except asyncio.CancelledError:
        await port.hangup(call_id, "session_cancelled")
        await runner.cancel(reason="session_cancelled")
        raise
    finally:
        if not terminal_task.done():
            terminal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await terminal_task


async def _wait_for_answer(events: AsyncIterator[CallEvent]) -> CallEvent:
    async for event in events:
        if event.event == CallEventType.ANSWERED:
            return event
        if event.event in {CallEventType.HANGUP, CallEventType.FAILED}:
            return event
    raise RuntimeError("call event stream ended before answer")


async def _wait_for_terminal(events: AsyncIterator[CallEvent]) -> CallEvent:
    async for event in events:
        if event.event in {CallEventType.HANGUP, CallEventType.FAILED}:
            return event
    raise RuntimeError("call event stream ended without terminal event")


def _terminal_result(event: CallEvent) -> SessionResult:
    result = (
        SessionResultType.CALL_FAILED
        if event.event == CallEventType.FAILED
        else SessionResultType.REMOTE_HANGUP
    )
    return SessionResult(
        call_id=event.call_id,
        result=result,
        reason=event.reason,
        sip_status=event.sip_status,
    )

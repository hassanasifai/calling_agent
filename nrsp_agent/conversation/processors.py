"""Pipecat processors that let a profile-driven engine run a call."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    EndWorkerFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService

from nrsp_agent.agent.engine import CallEngine, Decision
from nrsp_agent.agent.textnorm import is_degenerate, normalize

# In-call memory. The whole call is kept (calls are short; the engine caps
# turns): a sliding window would change the prompt prefix every turn and
# defeat the server's prefix cache (measured: 300-430 tokens re-processed,
# +7-11 s per turn on CPU).
HISTORY_MESSAGES = 64


@dataclass
class Turn:
    role: str
    text: str
    at_ms: int
    source: str


@dataclass
class ConversationRecord:
    """Per-call transcript and pipeline metrics, owned by one session."""

    started: float = field(default_factory=time.monotonic)
    turns: list[Turn] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    metrics: list[dict[str, object]] = field(default_factory=list)
    guard_rejections: list[str] = field(default_factory=list)

    def now_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def add_turn(self, role: str, text: str, source: str) -> None:
        self.turns.append(Turn(role, text, self.now_ms(), source))

    def add_event(self, name: str) -> None:
        self.events.append((self.now_ms(), name))


class MetricsCollector(BaseObserver):
    """Keep service TTFB/processing metrics for the call record."""

    def __init__(self, record: ConversationRecord) -> None:
        super().__init__()
        self._record = record

    async def on_push_frame(self, data: FramePushed) -> None:
        if not isinstance(data.frame, MetricsFrame) or not data.first_push:
            return
        for item in data.frame.data:
            if isinstance(item, TTFBMetricsData | ProcessingMetricsData):
                self._record.metrics.append(
                    {
                        "at_ms": self._record.now_ms(),
                        "kind": (
                            "ttfb"
                            if isinstance(item, TTFBMetricsData)
                            else "processing"
                        ),
                        "processor": item.processor,
                        "seconds": round(item.value, 4),
                    }
                )


def _last_user_text(context: LLMContext) -> str:
    for message in reversed(context.get_messages()):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
    return ""


class ConversationController(FrameProcessor):
    """Send each caller turn to the LLM with the engine's context.

    Sits between the user context aggregator and the LLM. The model answers
    with constrained JSON (action, value, say); :class:`LLMOutputGuard` hands
    the action back to :meth:`resolve` and the generic :class:`CallEngine`
    decides what is spoken. If no action arrives within
    ``understanding_timeout_s`` the engine's repeat request is spoken so the
    caller is never left in silence.
    """

    def __init__(
        self,
        engine: CallEngine,
        record: ConversationRecord,
        understanding_timeout_s: float = 8.0,
        finish_on_silence: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine
        self._record = record
        self._timeout_s = understanding_timeout_s
        self._finish_on_silence = finish_on_silence
        self._history: list[dict] = []
        self._turn_text = ""
        self._turn_open = False
        self._watchdog: asyncio.Task | None = None
        self._keys = ""
        # One LLM request in flight at a time; responses carry its id.
        self.request_id = 0
        self._awaiting_response = False

    def system_message(self) -> dict:
        return {"role": "system", "content": self.engine.system_prompt()}

    async def greet(self) -> None:
        decision = self.engine.greeting()
        self.assistant_said(decision.speak, "script", "greeting")
        await self.push_frame(TTSSpeakFrame(decision.speak))

    async def on_keypad(self, key: str) -> None:
        """One DTMF key. Digits are buffered while a verification is open and
        checked when complete (or on '#'); key values are never logged."""
        expected = self.engine.keypad_length()
        if expected is None:
            self._keys = ""
            return
        if key == "*":
            self._keys = ""
            return
        if key != "#":
            self._keys += key
        if key == "#" or len(self._keys) >= expected:
            digits, self._keys = self._keys, ""
            self._record.add_event("keypad_entry")
            # The keypad answer supersedes any spoken turn in flight.
            self._turn_open = False
            self._cancel_watchdog()
            await self._apply(self.engine.on_keypad(digits))

    async def handle_silence(self) -> None:
        self._record.add_event("user_silence_timeout")
        await self._apply(self.engine.on_silence(self._finish_on_silence))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            await self._on_user_turn(_last_user_text(frame.context))
            return
        await self.push_frame(frame, direction)

    async def _on_user_turn(self, text: str) -> None:
        self._record.add_turn("user", text, "asr")
        if self.engine.closed:
            return
        if is_degenerate(text):
            # An ASR repetition loop is noise, never meaning.
            self._record.add_event("asr_loop")
            await self._apply(self.engine.on_failure())
            return
        turn = {"role": "user", "content": self.engine.turn_message(text, self.redact)}
        messages = [self.system_message(), *self._history[-HISTORY_MESSAGES:], turn]
        # History keeps exactly what was sent (and below, exactly what the
        # model generated) so the next request shares the server's cached
        # prefix and only the new turn's tokens are processed.
        self._history.append(turn)
        if self._awaiting_response:
            # A slow answer to an earlier turn is still streaming: cancel it so
            # it can never be applied to this turn.
            self._record.add_event("llm_request_superseded")
            await self.push_frame(InterruptionFrame())
        self._turn_text, self._turn_open = text, True
        self.request_id += 1
        self._awaiting_response = True
        self._cancel_watchdog()
        self._watchdog = asyncio.create_task(self._fallback_after_timeout())
        await self.push_frame(LLMContextFrame(context=LLMContext(messages=messages)))

    def response_done(self) -> None:
        self._awaiting_response = False

    async def resolve(
        self, action: str | None, value: str, request_id: int
    ) -> Decision | None:
        """Called by the guard with the model's action (``None`` if unusable).

        Answers to an earlier request, or arriving after the turn timed out,
        are ignored."""
        if not self._turn_open or request_id != self.request_id:
            self._record.add_event("stale_model_output_dropped")
            return None
        self._turn_open = False
        self._cancel_watchdog()
        if action is None:
            self._record.add_event("model_output_invalid")
            decision = self.engine.on_failure()
        else:
            self._record.add_event(f"action:{action}")
            decision = self.engine.on_model(action, value, self._turn_text)
        await self._switch_asr(decision)
        if decision and decision.end_call:
            self._record.add_event(f"end_call:{self.engine.outcome}")
        return decision

    def assistant_said(
        self, text: str, source: str, action: str = "reply", raw: str = ""
    ) -> None:
        """Record what was spoken; ``raw`` is the model's exact output."""
        self._record.add_turn("assistant", text, source)
        said = {"action": action, "value": "", "say": self.redact(text)}
        content = raw or json.dumps(said, ensure_ascii=False)
        self._history.append({"role": "assistant", "content": content})

    def redact(self, text: str) -> str:
        # Private call data never enters the model's context.
        for secret in self.engine.secrets():
            text = text.replace(secret, "[private]")
        return text

    async def _fallback_after_timeout(self) -> None:
        await asyncio.sleep(self._timeout_s)
        if not self._turn_open:
            return
        self._turn_open = False
        self._record.add_event("model_timeout")
        logger.warning("LLM did not answer in time; asking the caller to repeat")
        await self._apply(self.engine.on_failure())

    def _cancel_watchdog(self) -> None:
        if (
            self._watchdog
            and not self._watchdog.done()
            and self._watchdog is not asyncio.current_task()
        ):
            self._watchdog.cancel()
        self._watchdog = None

    async def _switch_asr(self, decision: Decision | None) -> None:
        if decision is None or decision.asr_prompt is None:
            return
        # STT is upstream of this controller; the change applies to the
        # caller's next answer.
        await self.push_frame(
            STTUpdateSettingsFrame(
                delta=WhisperSTTService.Settings(initial_prompt=decision.asr_prompt)
            ),
            FrameDirection.UPSTREAM,
        )

    async def _apply(self, decision: Decision | None) -> None:
        """Speak an approved decision without the LLM (silence, failures)."""
        if decision is None:
            return
        await self._switch_asr(decision)
        self.assistant_said(decision.speak, "script")
        await self.push_frame(TTSSpeakFrame(decision.speak))
        if decision.end_call:
            self._record.add_event(f"end_call:{self.engine.outcome}")
            await self.push_frame(EndWorkerFrame(reason=self.engine.outcome))

    async def cleanup(self) -> None:
        self._cancel_watchdog()
        await super().cleanup()


_SENTENCE_END = re.compile(r"(?<=[۔؟!?.\n])")
_MARKUP = re.compile(r"[*#_`~>|\[\]{}]")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


def _words(text: str) -> set[str]:
    return set(normalize(text).split())


def is_echo(sentence: str, user_text: str, threshold: float = 0.6) -> bool:
    """Whether a reply mostly parrots the caller's (often mis-heard) words."""
    reply, heard = _words(sentence), _words(user_text)
    return bool(reply) and len(reply & heard) / len(reply) >= threshold


def screen_sentence(
    text: str, forbidden: tuple[str, ...] = (), allow_digits: bool = False
) -> str | None:
    """Return a speakable sentence, or ``None`` if it may invent facts.

    Numbers are rejected unless the profile allows them (facts come from
    approved texts); profile-forbidden terms (commitments the business has
    not authorised) are rejected; non-Urdu script is rejected because the
    Urdu voice cannot speak it intelligibly.
    """
    cleaned = "".join(ch for ch in text if ord(ch) <= 0xFFFF)
    cleaned = " ".join(_MARKUP.sub(" ", cleaned).split())
    if not cleaned or not any("\u0600" <= ch <= "\u06ff" for ch in cleaned):
        return None
    if not allow_digits and any(ch.isdigit() for ch in cleaned):
        return None
    if _DEVANAGARI.search(cleaned) or len(_LATIN.findall(cleaned)) > 3:
        return None
    if any(word in cleaned for word in forbidden):
        return None
    return cleaned


_ACTION = re.compile(r'"action"\s*:\s*"([a-z_]+)"')
_VALUE = re.compile(r'"value"\s*:\s*"((?:[^"\\]|\\.)*)"')
_SAY = re.compile(r'"say"\s*:\s*"')
_STRING_END = re.compile(r'(?<!\\)(?:\\\\)*"')


def _unescape(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw.replace('\\"', '"').replace("\\n", " ")


class LLMOutputGuard(FrameProcessor):
    """Interpret and screen the model's streamed JSON before it reaches TTS.

    As soon as ``action`` and ``value`` are complete the controller resolves
    the turn: an approved text is spoken at once and the rest of the model
    output is dropped; otherwise the ``say`` string is streamed sentence by
    sentence through :func:`screen_sentence`, then the approved follow-up.
    Without a controller (tests, plain chat) every text is treated as a reply.
    """

    def __init__(
        self,
        record: ConversationRecord,
        controller: ConversationController | None = None,
        max_sentences: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._record = record
        self._controller = controller
        self._max_sentences = max_sentences
        profile = controller.engine.profile if controller else None
        self._forbidden = profile.reply_forbidden_terms if profile else ()
        self._allow_digits = profile.reply_allow_digits if profile else False
        self._reset()

    def _reset(self) -> None:
        self._active = False
        self._request = 0
        self._raw = ""
        self._say_from: int | None = None
        self._say_done = False
        self._pending = ""
        self._spoken: list[str] = []
        self._action = "reply"
        self._resolved = self._controller is None
        self._decision: Decision | None = (
            Decision(use_model_reply=True) if self._controller is None else None
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            self._active = True
            if self._controller is not None:
                self._request = self._controller.request_id
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            await self._on_text(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._active or self._controller is None:
                await self._finish()
            self._active = False
            if self._controller is not None:
                self._controller.response_done()
            await self.push_frame(frame, direction)
            if self._decision is not None and self._decision.end_call:
                await self.push_frame(EndWorkerFrame(reason="end_call"))
        elif isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _on_text(self, text: str) -> None:
        if self._controller is None:
            self._pending += text
            await self._flush_sentences()
            return
        self._raw += text
        if not self._resolved:
            # Resolve as soon as the action name is complete; wait for the
            # value only when the action records it (saves ~0.5 s on CPU).
            action = _ACTION.search(self._raw)
            if not action:
                return
            name = action.group(1)
            value = _VALUE.search(self._raw)
            if self._controller.engine.needs_value(name) and not value:
                return
            await self._resolve(name, _unescape(value.group(1)) if value else "")
        if self._decision is None or not self._decision.use_model_reply:
            return
        if self._say_from is None:
            start = _SAY.search(self._raw)
            if not start:
                return
            self._say_from = start.end()
        if self._say_done:
            return
        segment = self._raw[self._say_from :]
        end = _STRING_END.search(segment)
        if end:
            segment, self._say_done = segment[: end.end() - 1], True
        self._say_from += len(segment)
        self._pending += segment
        await self._flush_sentences()

    async def _flush_sentences(self) -> None:
        parts = _SENTENCE_END.split(self._pending)
        self._pending = parts.pop()
        for sentence in parts:
            await self._emit(_unescape(sentence))

    async def _resolve(self, action: str | None, value: str) -> None:
        self._resolved = True
        self._action = action or "invalid"
        assert self._controller is not None
        self._decision = await self._controller.resolve(action, value, self._request)
        if self._decision is not None and self._decision.speak:
            await self._say(self._decision.speak)

    async def _finish(self) -> None:
        if not self._resolved:
            await self._resolve(None, "")
        decision = self._decision
        if decision is None:
            return
        if not decision.use_model_reply:
            assert self._controller is not None
            raw = self._raw.strip() if self._action != "invalid" else ""
            self._controller.assistant_said(decision.speak, "script", self._action, raw)
            return
        await self._emit(_unescape(self._pending))
        self._pending = ""
        model_said = " ".join(self._spoken)
        if decision.append:
            await self._say(decision.append)
        elif not self._spoken and self._controller is not None:
            # Never leave the caller in silence.
            engine = self._controller.engine
            await self._say(engine.render(engine.profile.repeat_request))
        text = " ".join(self._spoken)
        if self._controller is not None:
            self._controller.assistant_said(
                text, "llm", self._action, self._raw.strip()
            )
        else:
            self._record.add_turn("assistant", text, "llm")
        if not model_said:
            self._record.add_event("model_reply_empty_or_rejected")

    async def _say(self, text: str) -> None:
        self._spoken.append(text)
        await self.push_frame(LLMTextFrame(text=text + " "))

    async def _emit(self, sentence: str) -> None:
        if not sentence.strip() or len(self._spoken) >= self._max_sentences:
            return
        safe = screen_sentence(sentence, self._forbidden, self._allow_digits)
        if safe is not None and is_echo(safe, self._last_user_text()):
            safe = None
        if safe is None:
            self._record.guard_rejections.append(sentence.strip())
            logger.warning("LLM sentence rejected by output guard")
            return
        await self._say(safe)

    def _last_user_text(self) -> str:
        for turn in reversed(self._record.turns):
            if turn.role == "user":
                return turn.text
        return ""

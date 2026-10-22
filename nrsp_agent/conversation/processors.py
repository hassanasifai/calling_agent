"""Pipecat processors that put the deterministic policy in charge of a call."""

from __future__ import annotations

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

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import (
    AsrMode,
    ConversationPolicy,
    Decision,
    normalize,
)


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
    """Route each completed user turn through :class:`ConversationPolicy`.

    Sits between the user context aggregator and the LLM. Scripted replies are
    spoken directly; only turns the policy hands to the LLM reach it.
    """

    def __init__(
        self, policy: ConversationPolicy, record: ConversationRecord, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self._policy = policy
        self._record = record

    async def greet(self) -> None:
        text = self._policy.greeting()
        self._record.add_turn("assistant", text, "script")
        await self.push_frame(TTSSpeakFrame(text))

    async def handle_silence(self) -> None:
        self._record.add_event("user_silence_timeout")
        await self._apply(self._policy.on_silence(), "")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            text = _last_user_text(frame.context)
            self._record.add_turn("user", text, "asr")
            await self._apply(self._policy.on_user_turn(text), text)
            return
        await self.push_frame(frame, direction)

    async def _apply(self, decision: Decision | None, text: str) -> None:
        if decision is None:
            return
        if decision.asr_mode is not None:
            prompt = (
                script.ASR_PROMPT_DIGITS
                if decision.asr_mode == AsrMode.DIGITS
                else script.ASR_PROMPT_DOMAIN
            )
            # STT is upstream of this controller; the change applies to the
            # caller's next answer.
            await self.push_frame(
                STTUpdateSettingsFrame(
                    delta=WhisperSTTService.Settings(initial_prompt=prompt)
                ),
                FrameDirection.UPSTREAM,
            )
        if decision.use_llm and text:
            # A fresh minimal context: the LLM sees only an example and the
            # caller's last turn, never earlier turns containing customer facts.
            messages = [*script.DISCUSS_EXAMPLE, {"role": "user", "content": text}]
            await self.push_frame(
                LLMContextFrame(context=LLMContext(messages=messages))
            )
        elif decision.speak:
            self._record.add_turn("assistant", decision.speak, "script")
            await self.push_frame(TTSSpeakFrame(decision.speak))
        if decision.end_call:
            self._record.add_event(f"end_call:{self._policy.outcome.value}")
            await self.push_frame(EndWorkerFrame(reason=self._policy.outcome.value))


_SENTENCE_END = re.compile(r"(?<=[۔؟!?.\n])")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_MARKUP = re.compile(r"[*#_`~>|\[\]{}]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_FORBIDDEN_WORDS = (
    # Money and quantities: figures come only from verified templates.
    "ہزار",
    "لاکھ",
    "کروڑ",
    "روپے",
    "روپیہ",
    "روپیے",
    "فیصد",
    # Commitments the business has not authorised (qwen3:1.7b was measured
    # promising an extension).
    "مہلت",
    "توسیع",
    "رعایت",
    "معاف",
    "جرمانہ",
    "سہولت",
)


def _words(text: str) -> set[str]:
    return set(normalize(text).split())


def is_echo(sentence: str, user_text: str, threshold: float = 0.6) -> bool:
    """Whether a reply mostly parrots the caller's (often mis-heard) words."""
    reply, heard = _words(sentence), _words(user_text)
    return bool(reply) and len(reply & heard) / len(reply) >= threshold


def screen_sentence(text: str) -> str | None:
    """Return a speakable sentence, or ``None`` if it may invent facts.

    The LLM is not allowed to state numbers, money or dates at all; those come
    only from the verified templates. Non-Urdu script is rejected because the
    Urdu voice cannot speak it intelligibly.
    """
    cleaned = "".join(ch for ch in _THINK.sub(" ", text) if ord(ch) <= 0xFFFF)
    cleaned = " ".join(_MARKUP.sub(" ", cleaned).split())
    if not cleaned or not any("؀" <= ch <= "ۿ" for ch in cleaned):
        return None
    if any(ch.isdigit() for ch in cleaned) or _DEVANAGARI.search(cleaned):
        return None
    if len(_LATIN.findall(cleaned)) > 3:
        return None
    if any(word in cleaned for word in _FORBIDDEN_WORDS):
        return None
    return cleaned


class LLMOutputGuard(FrameProcessor):
    """Screen LLM text sentence by sentence before it reaches TTS."""

    def __init__(
        self, record: ConversationRecord, max_sentences: int = 2, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self._record = record
        self._max_sentences = max_sentences
        self._buffer = ""
        self._spoken: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer, self._spoken = "", []
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
            parts = _SENTENCE_END.split(self._buffer)
            self._buffer = parts.pop()
            for sentence in parts:
                await self._emit(sentence)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._emit(self._buffer)
            self._buffer = ""
            if not self._spoken:
                self._spoken.append(script.LLM_FALLBACK)
                await self.push_frame(LLMTextFrame(text=script.LLM_FALLBACK))
            self._record.add_turn("assistant", " ".join(self._spoken), "llm")
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _emit(self, sentence: str) -> None:
        if not sentence.strip() or len(self._spoken) >= self._max_sentences:
            return
        safe = screen_sentence(sentence)
        if safe is not None and is_echo(safe, self._last_user_text()):
            safe = None
        if safe is None:
            self._record.guard_rejections.append(sentence.strip())
            logger.warning("LLM sentence rejected by output guard")
            return
        self._spoken.append(safe)
        await self.push_frame(LLMTextFrame(text=safe + " "))

    def _last_user_text(self) -> str:
        for turn in reversed(self._record.turns):
            if turn.role == "user":
                return turn.text
        return ""

from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    EndWorkerFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import ConversationPolicy, Outcome
from nrsp_agent.conversation.processors import (
    ConversationController,
    ConversationRecord,
    LLMOutputGuard,
    is_echo,
    screen_sentence,
)
from nrsp_agent.devtools.loopback import TEST_CUSTOMER as CUSTOMER


@pytest.mark.parametrize(
    "sentence",
    [
        "آپ کی قسط 5000 روپے ہے۔",
        "آپ کی قسط ۵۰۰۰ ہے۔",
        "آپ کی پانچ ہزار روپے کی قسط ہے۔",
        "ہم آپ کو مہلت دیں گے۔",
        "Sure, I can help.",
        "मैं आपकी मदद करूँगा।",
        "",
    ],
)
def test_guard_rejects_invented_facts_and_wrong_script(sentence: str) -> None:
    assert screen_sentence(sentence) is None


def test_guard_cleans_markup_and_keeps_urdu() -> None:
    assert screen_sentence("**بہت شکریہ!** 😊") == "بہت شکریہ!"


def test_echo_detection() -> None:
    heard = "جی ہاں، میں ابھی لے حافظ کے دبکہ عدا کر دوں گیا"
    assert is_echo("ہاں، میں ابھی لے حافظ کے دبکہ عدا کر دوں گیا۔", heard)
    assert not is_echo("بہت شکریہ، ہم آپ کے تعاون کے شکر گزار ہیں۔", heard)


class Capture:
    def __init__(self, processor) -> None:
        self.frames: list = []

        async def push(frame, direction=FrameDirection.DOWNSTREAM):
            self.frames.append(frame)

        processor.push_frame = push

    def of(self, kind):
        return [f for f in self.frames if isinstance(f, kind)]


async def stream_llm(guard: LLMOutputGuard, chunks: list[str]) -> None:
    await guard.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for chunk in chunks:
        await guard.process_frame(LLMTextFrame(text=chunk), FrameDirection.DOWNSTREAM)
    await guard.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


@pytest.mark.asyncio
async def test_guard_streams_sentences_and_drops_unsafe_ones() -> None:
    record = ConversationRecord()
    guard = LLMOutputGuard(record)
    out = Capture(guard)

    await stream_llm(guard, ["بہت شکر", "یہ۔ آپ کی قسط 5000", " ہے۔ کیا کچھ اور؟"])

    texts = [f.text.strip() for f in out.of(LLMTextFrame)]
    assert texts == ["بہت شکریہ۔", "کیا کچھ اور؟"]
    assert record.guard_rejections == ["آپ کی قسط 5000 ہے۔"]
    assert record.turns[-1].source == "llm"


@pytest.mark.asyncio
async def test_guard_falls_back_when_everything_is_rejected() -> None:
    guard = LLMOutputGuard(ConversationRecord())
    out = Capture(guard)

    await stream_llm(guard, ["Okay, I will help you."])

    assert [f.text for f in out.of(LLMTextFrame)] == [script.LLM_FALLBACK]


@pytest.mark.asyncio
async def test_guard_limits_sentence_count() -> None:
    guard = LLMOutputGuard(ConversationRecord(), max_sentences=2)
    out = Capture(guard)

    await stream_llm(guard, ["پہلا جملہ۔ دوسرا جملہ۔ تیسرا جملہ۔"])

    assert len(out.of(LLMTextFrame)) == 2


def user_turn(text: str) -> LLMContextFrame:
    return LLMContextFrame(
        context=LLMContext(messages=[{"role": "user", "content": text}])
    )


@pytest.mark.asyncio
async def test_controller_speaks_scripts_and_never_forwards_unverified_turns() -> None:
    policy = ConversationPolicy(CUSTOMER)
    controller = ConversationController(policy, ConversationRecord())
    out = Capture(controller)

    await controller.process_frame(user_turn("جی ہاں"), FrameDirection.DOWNSTREAM)

    assert out.of(LLMContextFrame) == []
    assert out.of(TTSSpeakFrame)[0].text.endswith(script.ASK_CNIC_LAST4)


@pytest.mark.asyncio
async def test_controller_gives_llm_only_the_last_turn() -> None:
    policy = ConversationPolicy(CUSTOMER)
    controller = ConversationController(policy, ConversationRecord())
    out = Capture(controller)
    for text in ("جی ہاں", "چار پانچ چھ سات"):
        await controller.process_frame(user_turn(text), FrameDirection.DOWNSTREAM)

    history = LLMContext(
        messages=[
            {"role": "assistant", "content": "آپ کی پانچ ہزار روپے کی قسط"},
            {"role": "user", "content": "جی میں کوشش کروں گا"},
        ]
    )
    await controller.process_frame(
        LLMContextFrame(context=history), FrameDirection.DOWNSTREAM
    )

    sent = out.of(LLMContextFrame)[-1].context.get_messages()
    assert sent[-1] == {"role": "user", "content": "جی میں کوشش کروں گا"}
    assert all("ہزار" not in str(m.get("content")) for m in sent)


@pytest.mark.asyncio
async def test_controller_ends_call_after_closing_speech() -> None:
    policy = ConversationPolicy(CUSTOMER)
    record = ConversationRecord()
    controller = ConversationController(policy, record)
    out = Capture(controller)

    await controller.process_frame(
        user_turn("دوبارہ کال نہ کریں"), FrameDirection.DOWNSTREAM
    )

    kinds = [type(f) for f in out.frames]
    assert kinds == [TTSSpeakFrame, EndWorkerFrame]
    assert out.frames[1].reason == Outcome.OPTED_OUT.value
    assert "end_call:opted_out" in [e for _, e in record.events]


@pytest.mark.asyncio
async def test_controller_greets_once_per_call_record() -> None:
    record = ConversationRecord()
    controller = ConversationController(ConversationPolicy(CUSTOMER), record)
    out = Capture(controller)

    await controller.greet()

    assert len(out.of(TTSSpeakFrame)) == 1
    assert [t.source for t in record.turns] == ["script"]


@pytest.mark.asyncio
async def test_controller_scopes_digit_asr_prompt_to_verification() -> None:
    from pipecat.frames.frames import STTUpdateSettingsFrame

    controller = ConversationController(
        ConversationPolicy(CUSTOMER), ConversationRecord()
    )
    out = Capture(controller)

    await controller.process_frame(user_turn("جی ہاں"), FrameDirection.DOWNSTREAM)
    await controller.process_frame(
        user_turn("چار پانچ چھ سات"), FrameDirection.DOWNSTREAM
    )

    prompts = [f.delta.initial_prompt for f in out.of(STTUpdateSettingsFrame)]
    assert prompts == [script.ASR_PROMPT_DIGITS, script.ASR_PROMPT_DOMAIN]
    assert CUSTOMER.cnic_last4 not in script.ASR_PROMPT_DIGITS

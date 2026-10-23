from __future__ import annotations

import asyncio
import json

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

from nrsp_agent.agent import build_engine
from nrsp_agent.conversation.processors import (
    ConversationController,
    ConversationRecord,
    LLMOutputGuard,
    is_echo,
    screen_sentence,
)
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA

FORBIDDEN = ("ہزار", "روپے", "مہلت")


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
    assert screen_sentence(sentence, FORBIDDEN) is None


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
async def test_guard_limits_sentence_count() -> None:
    guard = LLMOutputGuard(ConversationRecord(), max_sentences=2)
    out = Capture(guard)

    await stream_llm(guard, ["پہلا جملہ۔ دوسرا جملہ۔ تیسرا جملہ۔"])

    assert len(out.of(LLMTextFrame)) == 2


def user_turn(text: str) -> LLMContextFrame:
    return LLMContextFrame(
        context=LLMContext(messages=[{"role": "user", "content": text}])
    )


def model_json(action: str, say: str = "", value: str = "") -> str:
    return json.dumps(
        {"action": action, "value": value, "say": say}, ensure_ascii=False
    )


class Call:
    """Controller + guard wired as in the pipeline, with a scripted model."""

    def __init__(self, timeout_s: float = 6.0, prefill=None) -> None:
        self.engine = build_engine(NRSP_PROFILE, TEST_CALL_DATA)
        self.record = ConversationRecord()
        self.controller = ConversationController(
            self.engine, self.record, understanding_timeout_s=timeout_s, prefill=prefill
        )
        self.guard = LLMOutputGuard(self.record, self.controller)
        self.up = Capture(self.controller)
        self.out = Capture(self.guard)

    async def turn(self, caller: str, model_output: str | None) -> list[str]:
        """Return the texts spoken for this turn (streamed in small chunks)."""
        before = len(self.out.frames)
        await self.controller.process_frame(
            user_turn(caller), FrameDirection.DOWNSTREAM
        )
        if model_output is not None:
            chunks = [model_output[i : i + 5] for i in range(0, len(model_output), 5)]
            await stream_llm(self.guard, chunks)
        # Approved texts go to TTS as TTSSpeakFrame, model replies as LLM text.
        return [
            f.text.strip()
            for f in self.out.frames[before:]
            if isinstance(f, LLMTextFrame | TTSSpeakFrame)
        ]

    def sent_messages(self) -> list[dict]:
        return self.up.of(LLMContextFrame)[-1].context.get_messages()

    async def verify(self) -> None:
        await self.turn("جی", model_json("confirm_identity"))
        await self.turn("چار پانچ چھ سات", model_json("verify_cnic"))
        await self.turn("جی درست", model_json("affirm"))
        assert "verified" in self.engine.flags


@pytest.mark.asyncio
async def test_every_turn_goes_to_the_llm_with_engine_context() -> None:
    call = Call()
    spoken = await call.turn("haan ji main hi hoon", model_json("confirm_identity"))

    messages = call.sent_messages()
    assert messages[0]["role"] == "system" and "NRSP" in messages[0]["content"]
    assert "you asked:" in messages[-1]["content"]
    assert "haan ji main hi hoon" in messages[-1]["content"]
    assert spoken == [call.engine.render("شکریہ۔ {ask_cnic}")]


@pytest.mark.asyncio
async def test_end_action_speaks_approved_text_and_drops_model_text() -> None:
    call = Call()
    confirm = await call.turn("آئندہ فون مت کیجیے گا", model_json("opt_out"))
    assert confirm[0].endswith("کال نہ کرے؟")
    spoken = await call.turn("جی بالکل", model_json("affirm", "کچھ بھی"))

    assert len(spoken) == 1 and "درخواست نوٹ" in spoken[0]
    assert isinstance(call.out.frames[-1], EndWorkerFrame)
    assert call.engine.outcome == "opted_out"
    assert "end_call:opted_out" in [e for _, e in call.record.events]


@pytest.mark.asyncio
async def test_model_reply_is_guarded_and_followed_by_pending_question() -> None:
    call = Call()
    await call.verify()

    spoken = await call.turn(
        "قسط کہاں جمع ہو گی؟",
        model_json(
            "reply", "ہمارا نمائندہ آپ کو تفصیل بتائے گا۔ ہم آپ کو مہلت دیں گے۔"
        ),
    )

    assert spoken == [
        "ہمارا نمائندہ آپ کو تفصیل بتائے گا۔",
        "کیا آپ مقررہ تاریخ تک ادائیگی کر سکیں گے؟",
    ]
    assert call.record.guard_rejections == ["ہم آپ کو مہلت دیں گے۔"]


@pytest.mark.asyncio
async def test_history_never_contains_private_call_data() -> None:
    call = Call()
    await call.verify()
    await call.turn("اچھا", model_json("reply", "جی بہتر۔"))

    sent = str(call.sent_messages())
    for secret in call.engine.secrets():
        assert secret not in sent
    assert "installment_disclosed" in sent  # the model knows it happened


@pytest.mark.asyncio
async def test_invalid_output_asks_caller_to_repeat() -> None:
    call = Call()
    spoken = await call.turn("ہیلو", "I think the caller wants")

    assert spoken[0].startswith("معذرت، میں آپ کی بات")
    assert "model_output_invalid" in [e for _, e in call.record.events]


@pytest.mark.asyncio
async def test_llm_timeout_asks_to_repeat_and_late_output_is_ignored() -> None:
    import asyncio

    call = Call(timeout_s=0.01)
    await call.turn("جی ہاں", None)
    await asyncio.sleep(0.05)

    assert call.up.of(TTSSpeakFrame)[-1].text.startswith("معذرت")
    before = len(call.out.frames)
    await stream_llm(call.guard, [model_json("opt_out")])
    assert not any(isinstance(f, LLMTextFrame) for f in call.out.frames[before:])
    assert call.engine.outcome == "in_progress"


@pytest.mark.asyncio
async def test_closed_call_ignores_further_turns() -> None:
    call = Call()
    await call.turn("نہیں یہ غلط نمبر ہے", model_json("wrong_person"))
    await call.turn("جی", model_json("affirm"))
    assert call.engine.outcome == "wrong_person"
    sent_before = len(call.up.of(LLMContextFrame))

    await call.turn("ہیلو", None)

    assert len(call.up.of(LLMContextFrame)) == sent_before


@pytest.mark.asyncio
async def test_controller_greets_once_per_call_record() -> None:
    call = Call()
    await call.controller.greet()

    assert len(call.up.of(TTSSpeakFrame)) == 1
    assert [t.source for t in call.record.turns] == ["script"]


@pytest.mark.asyncio
async def test_asr_prompt_follows_profile_actions() -> None:
    from pipecat.frames.frames import STTUpdateSettingsFrame

    call = Call()
    await call.verify()

    prompts = [f.delta.initial_prompt for f in call.up.of(STTUpdateSettingsFrame)]
    profile = call.engine.profile
    assert prompts == [profile.asr_prompts["digits"], profile.asr_prompts["default"]]
    assert TEST_CALL_DATA["cnic_last4"] not in profile.asr_prompts["digits"]


@pytest.mark.asyncio
async def test_keypad_digits_are_buffered_and_never_logged() -> None:
    call = Call()
    await call.turn("جی", model_json("confirm_identity"))
    for key in "45":
        await call.controller.on_keypad(key)
    assert "verified" not in call.engine.flags
    for key in "67":
        await call.controller.on_keypad(key)

    assert "verified" in call.engine.flags
    spoken = call.up.of(TTSSpeakFrame)[-1].text
    assert TEST_CALL_DATA["amount_spoken"] in spoken
    logged = str(call.record.events) + str(call.record.turns[:-1])
    assert "4567" not in logged and "keypad_entry" in logged


@pytest.mark.asyncio
async def test_answer_to_a_superseded_request_is_never_applied() -> None:
    from pipecat.frames.frames import InterruptionFrame

    call = Call(timeout_s=0.01)
    await call.controller.process_frame(user_turn("جی ہاں"), FrameDirection.DOWNSTREAM)
    down = FrameDirection.DOWNSTREAM
    await call.guard.process_frame(LLMFullResponseStartFrame(), down)  # request 1
    import asyncio

    await asyncio.sleep(0.05)  # turn 1 times out; caller speaks again
    await call.controller.process_frame(
        user_turn("میں فون پر نمبر نہیں بتاتا"), FrameDirection.DOWNSTREAM
    )
    assert any(isinstance(f, InterruptionFrame) for f in call.up.frames)
    # Request 1's late answer arrives while request 2 is open.
    await call.guard.process_frame(LLMTextFrame(text=model_json("wrong_person")), down)
    await call.guard.process_frame(LLMFullResponseEndFrame(), down)

    assert call.engine.pending_ask.startswith("کیا میری بات")  # nothing applied
    assert "stale_model_output_dropped" in [e for _, e in call.record.events]


@pytest.mark.asyncio
async def test_prefill_is_an_exact_prefix_of_the_next_request() -> None:
    prefills: list[list[dict]] = []

    async def prefill(messages: list[dict]) -> None:
        prefills.append(messages)

    call = Call(prefill=prefill)
    await call.controller.greet()
    await asyncio.sleep(0)
    await call.turn("جی میں ہی ہوں", model_json("confirm_identity"))
    await asyncio.sleep(0)
    sent = call.sent_messages()
    # The prefill after the greeting shares everything but the caller's words.
    first = prefills[0]
    assert first[:-1] == sent[:-1]
    assert sent[-1]["content"].startswith(first[-1]["content"])
    assert "جی میں ہی ہوں" not in first[-1]["content"]
    # After the model's answer the next prefill covers it too.
    await call.turn("چار پانچ چھ سات", model_json("verify_cnic"))
    assert prefills[1][:-1] == call.sent_messages()[:-1]

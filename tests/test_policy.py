from __future__ import annotations

import pytest

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import (
    ConversationPolicy,
    CustomerContext,
    Intent,
    Outcome,
    Stage,
    detect_intent,
    digits_match,
    extract_digits,
)

CUSTOMER = CustomerContext(
    customer_name="احمد علی",
    cnic_last4="4567",
    installment_amount_pkr=5000,
    installment_amount_spoken="پانچ ہزار",
    due_date_spoken="پندرہ نومبر",
)
SECRETS = ("پانچ ہزار", "پندرہ نومبر", "4567")


def spoken(decision) -> str:
    return decision.speak or ""


def test_greeting_discloses_ai_and_names_customer_only() -> None:
    greeting = ConversationPolicy(CUSTOMER).greeting()

    assert "اے آئی" in greeting
    assert "احمد علی" in greeting
    assert not any(secret in greeting for secret in SECRETS)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("مجھے دوبارہ کال نہ کریں", Intent.OPT_OUT),
        ("مجھے کسی نمائندے سے بات کرنی ہے", Intent.HANDOFF),
        ("ٹھیک ہے شکریہ، اللہ حافظ", Intent.GOODBYE),
        ("جی نہیں", Intent.NEGATIVE),
        ("جی ہاں میں بول رہا ہوں", Intent.AFFIRMATIVE),
        ("ابھی میرے پاس پیسے نہیں ہیں", Intent.PAYMENT_DIFFICULTY),
        ("آپ کون بول رہے ہیں؟", Intent.WHO_IS_CALLING),
        ("جیسے آپ کی مرضی", Intent.UNKNOWN),  # "جی" must match whole words only
        # Measured mis-hearing of "اگلے ہفتے تک": not a goodbye mid-sentence.
        ("جی ہاں، اللہ حافظ کے دبکہ عدا کر دوں میں", Intent.AFFIRMATIVE),
        ("ٹھیک ہے، کیوں کری ہے؟ اللہ حافظ", Intent.GOODBYE),
        ("خدا حافظ جی", Intent.GOODBYE),
    ],
)
def test_intents(text: str, intent: Intent) -> None:
    assert detect_intent(text) == intent


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("چار پانچ چھ سات", "4567"),
        ("آخری چار ہندسے چار پانچ چھ سات ہیں", "44567"),
        ("۴۵۶۷", "4567"),
        ("ہندس سے 4، 5، 6، 7", "4567"),
    ],
)
def test_extract_digits(text: str, digits: str) -> None:
    assert extract_digits(text) == digits


@pytest.mark.parametrize(
    ("heard", "ok"),
    [
        ("4567", True),
        ("44567", True),  # "آخری چار" heard as a digit
        ("45675", True),  # Whisper repeated a digit (measured)
        ("4576", False),
        ("456", False),
        ("0123456789", False),  # brute-force recitation is capped
        ("1234567", False),
    ],
)
def test_digits_match(heard: str, ok: bool) -> None:
    assert digits_match(heard, "4567") is ok


def test_repetition_loop_is_unclear_not_handoff() -> None:
    policy = ConversationPolicy(CUSTOMER)
    policy.on_user_turn("جی ہاں")
    policy.on_user_turn("چار پانچ چھ سات")

    decision = policy.on_user_turn("نمائندہ، نمائندہ، نمائندہ، نمائندہ")

    assert not decision.end_call and policy.outcome == Outcome.IN_PROGRESS
    assert spoken(decision).startswith(script.UNCLEAR)


def test_repeated_digit_is_not_degenerate() -> None:
    policy = ConversationPolicy(CUSTOMER.model_copy(update={"cnic_last4": "1111"}))
    policy.on_user_turn("جی ہاں")

    policy.on_user_turn("ایک ایک ایک ایک")

    assert policy.verified


def test_prompts_contain_no_intent_triggers() -> None:
    for prompt in (script.ASR_PROMPT_DOMAIN, script.ASR_PROMPT_DIGITS):
        assert detect_intent(prompt) not in {
            Intent.GOODBYE,
            Intent.HANDOFF,
            Intent.OPT_OUT,
            Intent.NEGATIVE,
            Intent.PAYMENT_DIFFICULTY,
        }


def test_happy_path_discloses_only_after_verification() -> None:
    policy = ConversationPolicy(CUSTOMER)

    ask = policy.on_user_turn("جی ہاں، میں بات کر رہا ہوں")
    assert policy.stage == Stage.VERIFY
    assert spoken(ask).endswith(script.ASK_CNIC_LAST4)
    assert not any(secret in spoken(ask) for secret in SECRETS[:2])

    notice = policy.on_user_turn("میرے آخری چار ہندسے چار پانچ چھ سات ہیں")
    assert policy.verified and policy.stage == Stage.DISCUSS
    assert "پانچ ہزار" in spoken(notice) and "پندرہ نومبر" in spoken(notice)

    assert policy.on_user_turn("جی میں اگلے ہفتے ادا کر دوں گا").use_llm
    closing = policy.on_user_turn("شکریہ اللہ حافظ")
    assert closing.end_call and policy.outcome == Outcome.COMPLETED
    assert policy.on_user_turn("ہیلو") is None  # nothing after closing


def test_wrong_digits_fail_closed_without_disclosure() -> None:
    policy = ConversationPolicy(CUSTOMER)
    policy.on_user_turn("جی ہاں")

    retry = policy.on_user_turn("ایک دو تین چار")
    assert not retry.end_call and spoken(retry).endswith(script.ASK_CNIC_LAST4)
    final = policy.on_user_turn("نو آٹھ سات چھ")

    assert final.end_call and policy.outcome == Outcome.VERIFICATION_FAILED
    assert not policy.verified
    assert not any(secret in spoken(final) for secret in SECRETS)


def test_too_few_digits_reasks_without_spending_an_attempt() -> None:
    policy = ConversationPolicy(CUSTOMER)
    policy.on_user_turn("جی ہاں")

    policy.on_user_turn("تین چار")  # mis-heard, only two digits
    policy.on_user_turn("ایک دو تین چار")  # first real failure
    assert policy.stage == Stage.VERIFY


@pytest.mark.parametrize(
    ("text", "outcome", "reply"),
    [
        ("نہیں، یہ غلط نمبر ہے", Outcome.WRONG_PERSON, script.WRONG_PERSON),
        ("مجھے دوبارہ کال نہ کریں", Outcome.OPTED_OUT, script.OPT_OUT),
        ("کسی نمائندے سے بات کرائیں", Outcome.HANDOFF_REQUESTED, script.HANDOFF),
    ],
)
def test_exits_before_verification(text: str, outcome: Outcome, reply: str) -> None:
    policy = ConversationPolicy(CUSTOMER)

    decision = policy.on_user_turn(text)

    assert decision.end_call and spoken(decision) == reply
    assert policy.outcome == outcome and not policy.verified


def test_payment_difficulty_is_scripted_not_llm() -> None:
    policy = ConversationPolicy(CUSTOMER)
    policy.on_user_turn("جی ہاں")
    policy.on_user_turn("چار پانچ چھ سات")

    decision = policy.on_user_turn("ابھی پیسے نہیں ہیں، کیا مہلت مل سکتی ہے؟")

    assert not decision.use_llm and spoken(decision) == script.PAYMENT_DIFFICULTY
    assert "payment_difficulty" in policy.flags


def test_silence_reprompts_then_closes() -> None:
    policy = ConversationPolicy(CUSTOMER, max_silence_prompts=2)

    first = policy.on_silence()
    assert spoken(first).startswith(script.SILENCE_PROMPT)
    assert "احمد علی" in spoken(first)  # repeats the pending question
    policy.on_silence()
    final = policy.on_silence()

    assert final.end_call and policy.outcome == Outcome.NO_RESPONSE
    assert policy.on_silence() is None


def test_speech_resets_silence_count() -> None:
    policy = ConversationPolicy(CUSTOMER, max_silence_prompts=1)
    policy.on_silence()
    policy.on_user_turn("کیا؟")  # unclear but the caller is present

    assert not policy.on_silence().end_call


def test_unclear_turns_close_after_limit() -> None:
    policy = ConversationPolicy(CUSTOMER, max_unclear_turns=1)

    assert not policy.on_user_turn("کیا؟").end_call
    final = policy.on_user_turn("ہیلو؟")

    assert final.end_call and policy.outcome == Outcome.NO_RESPONSE


def test_llm_turns_are_bounded() -> None:
    policy = ConversationPolicy(CUSTOMER, max_llm_turns=1)
    policy.on_user_turn("جی ہاں")
    policy.on_user_turn("چار پانچ چھ سات")

    assert policy.on_user_turn("جی میں کوشش کروں گا").use_llm
    final = policy.on_user_turn("اچھا میں دیکھتا ہوں")

    assert final.end_call and policy.outcome == Outcome.COMPLETED


def test_customer_context_rejects_bad_cnic_suffix() -> None:
    with pytest.raises(ValueError):
        CustomerContext(
            customer_name="x",
            cnic_last4="45a7",
            installment_amount_pkr=1,
            installment_amount_spoken="ایک",
            due_date_spoken="کل",
        )


def verified_policy() -> ConversationPolicy:
    policy = ConversationPolicy(CUSTOMER)
    policy.on_user_turn("جی ہاں")
    policy.on_user_turn("چار پانچ چھ سات")
    return policy


def test_no_after_anything_else_closes_completed() -> None:
    policy = verified_policy()
    assert policy.on_user_turn("جی میں وقت پر ادا کر دوں گا").use_llm

    final = policy.on_user_turn("نہیں شکریہ")

    assert final.end_call and policy.outcome == Outcome.COMPLETED


def test_bare_no_to_payment_question_is_difficulty_not_llm() -> None:
    policy = verified_policy()

    decision = policy.on_user_turn("نہیں")

    assert spoken(decision) == script.PAYMENT_DIFFICULTY and not decision.use_llm
    assert "payment_difficulty" in policy.flags


def test_silence_after_anything_else_closes_completed() -> None:
    policy = verified_policy()
    policy.on_user_turn("جی میں ادا کر دوں گا")

    final = policy.on_silence()

    assert final.end_call and policy.outcome == Outcome.COMPLETED


def test_measured_goodbye_misspelling() -> None:
    assert detect_intent("ٹھیک ہے، شکریہ، اللہ حافت") == Intent.GOODBYE

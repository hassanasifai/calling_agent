from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nrsp_agent.agent import build_engine
from nrsp_agent.agent.engine import AFFIRM, REJECT, CallEngine
from nrsp_agent.agent.knowledge import KnowledgeBase
from nrsp_agent.agent.profile import AgentProfile, load_profile
from nrsp_agent.agent.textnorm import extract_digits
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA

CLINIC = Path("tests/fixtures/profiles/clinic_reminder")
CLINIC_DATA = {
    "patient_name": "سارہ",
    "doctor": "ڈاکٹر عمران",
    "appointment_time": "منگل صبح دس بجے",
}


def nrsp() -> CallEngine:
    return build_engine(NRSP_PROFILE, TEST_CALL_DATA)


def verified_nrsp() -> CallEngine:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    engine.on_model("verify_cnic", "", "چار پانچ چھ سات")
    engine.on_model(AFFIRM, "", "جی درست")
    assert "verified" in engine.flags
    return engine


def test_private_data_never_in_model_context() -> None:
    engine = nrsp()
    context = engine.system_prompt() + engine.turn_message("hello", lambda s: s)
    for key in ("cnic_last4", "amount_spoken", "due_date_spoken"):
        assert TEST_CALL_DATA[key] not in context
    assert TEST_CALL_DATA["customer_name"] in context


def test_gated_disclosure_is_not_available_before_verification() -> None:
    engine = nrsp()
    decision = engine.on_model("tell_installment", "", "kitne paise?")

    assert decision.use_model_reply and not decision.speak
    assert TEST_CALL_DATA["amount_spoken"] not in decision.append


def test_verification_uses_parsed_digits_not_model_value() -> None:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    # The model claims the right digits, but the caller's words had none.
    decision = engine.on_model("verify_cnic", "4567", "جی ابھی بتاتا ہوں")

    assert "verified" not in engine.flags
    assert decision.speak.startswith("معذرت، براہِ کرم ہندسے")


def test_readback_then_confirm_verifies_and_discloses() -> None:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    readback = engine.on_model("verify_cnic", "", "آخری چار ہندسے چار پانچ چھ سات")

    assert "چار، پانچ، چھ، سات" in readback.speak
    assert "verified" not in engine.flags
    notice = engine.on_model(AFFIRM, "", "جی ہاں")
    assert TEST_CALL_DATA["amount_spoken"] in notice.speak
    assert engine.flags >= {"verified", "installment_disclosed"}


def test_rejected_readback_reasks_without_spending_an_attempt() -> None:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    engine.on_model("verify_cnic", "", "چار پانچ چھ آٹھ")
    again = engine.on_model(REJECT, "", "نہیں غلط")

    assert again.speak.endswith(engine.render("{ask_cnic}"))
    engine.on_model("verify_cnic", "", "چار پانچ چھ سات")
    engine.on_model(AFFIRM, "", "جی")
    assert "verified" in engine.flags


def test_wrong_digits_fail_closed_after_max_attempts() -> None:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    for _ in range(2):
        engine.on_model("verify_cnic", "", "ایک دو تین چار")
        decision = engine.on_model(AFFIRM, "", "جی")

    assert decision.end_call and engine.outcome == "verification_failed"
    assert TEST_CALL_DATA["amount_spoken"] not in decision.speak


def test_model_reply_returns_to_pending_question() -> None:
    engine = nrsp()
    decision = engine.on_model("reply", "", "aap kaun?")

    assert decision.use_model_reply
    assert decision.append == "کیا میری بات احمد علی سے ہو رہی ہے؟"


def test_record_action_stores_structured_field() -> None:
    engine = verified_nrsp()
    decision = engine.on_model("record_promise", "اگلے جمعہ", "جمعہ کو دے دوں گا")

    assert decision.use_model_reply and "کچھ اور" in decision.append
    assert engine.result()["fields"] == {"promise_to_pay": "اگلے جمعہ"}


@pytest.mark.parametrize(
    ("action", "outcome", "confirmed"),
    [
        ("opt_out", "opted_out", True),
        ("human_handoff", "handoff_requested", True),
        ("callback", "callback_requested", False),
        ("wrong_person", "wrong_person", True),
        ("finish", "caller_ended", False),
    ],
)
def test_end_actions_before_verification(
    action: str, outcome: str, confirmed: bool
) -> None:
    engine = nrsp()
    decision = engine.on_model(action, "", "...")
    if confirmed:  # profile asks before this irreversible ending
        assert not decision.end_call and decision.speak.endswith("؟")
        decision = engine.on_model(AFFIRM, "", "جی ہاں")

    assert decision.end_call and engine.outcome == outcome
    assert engine.on_model("reply", "", "hello") is None


def test_rejected_end_confirmation_returns_to_the_question() -> None:
    engine = nrsp()
    engine.on_model("wrong_person", "", "aap kaun?")
    back = engine.on_model(REJECT, "", "nahi nahi main hi hoon")

    assert not back.end_call and engine.outcome == "in_progress"
    assert back.speak == "کیا میری بات احمد علی سے ہو رہی ہے؟"
    assert AFFIRM not in engine.allowed_actions()


def test_finish_after_objective_is_completed() -> None:
    engine = verified_nrsp()
    assert engine.on_model("finish", "", "bas shukriya").end_call
    assert engine.outcome == "completed"


def test_silence_reprompts_then_closes_and_finishes_when_done() -> None:
    engine = nrsp()
    first = engine.on_silence("finish")
    assert first.speak.endswith("سے ہو رہی ہے؟") and not first.end_call
    engine.on_silence("finish")
    assert engine.on_silence("finish").end_call
    assert engine.outcome == "no_response"

    done = verified_nrsp()
    assert done.on_silence("finish").end_call and done.outcome == "completed"


def test_failures_ask_to_repeat_then_close() -> None:
    engine = nrsp()
    assert not engine.on_failure().end_call
    assert not engine.on_failure().end_call
    assert engine.on_failure().end_call and engine.outcome == "unclear"


def test_schema_allows_only_actions_available_this_turn() -> None:
    engine = nrsp()

    def enum() -> list[str]:
        schema = engine.response_schema()
        branches = schema.get("anyOf", [schema])
        return [n for b in branches for n in b["properties"]["action"]["enum"]]

    assert enum()[0] == "reply" and "confirm_identity" in enum()
    assert "verify_cnic" not in enum() and AFFIRM not in enum()
    assert "tell_installment" not in enum()  # gated until verified
    talking, silent = engine.response_schema()["anyOf"]
    assert list(talking["properties"]) == ["action", "value", "say"]
    assert silent["properties"]["say"]["maxLength"] == 0
    engine.on_model("confirm_identity", "", "جی")
    engine.on_model("verify_cnic", "", "چار پانچ چھ سات")
    # While the read-back awaits yes/no, only answers, a new attempt and
    # safety exits are possible.
    assert set(enum()) == {AFFIRM, REJECT, "verify_cnic", "opt_out", "human_handoff"}


def test_unrelated_action_during_confirmation_repeats_the_question() -> None:
    engine = nrsp()
    engine.on_model("confirm_identity", "", "جی")
    readback = engine.on_model("verify_cnic", "", "چار پانچ چھ سات")
    again = engine.on_model("finish", "", "جی")

    assert not again.end_call and again.speak.endswith(readback.speak)


def test_roman_urdu_digits_are_parsed() -> None:
    assert extract_digits("char paanch chay saat") == "4567"
    assert extract_digits("ek sifar nau aath") == "1098"


def test_second_profile_runs_on_the_same_engine() -> None:
    engine = build_engine(CLINIC, CLINIC_DATA)

    assert "City Clinic" in engine.system_prompt()
    assert "verify" not in engine.system_prompt().lower()
    slot = engine.on_model("identity_ok", "", "جی میں سارہ")
    assert "منگل صبح دس بجے" in slot.speak  # private value via approved text
    assert "منگل" not in engine.system_prompt()
    engine.on_model("reschedule", "جمعرات شام", "منگل نہیں آ سکتی")
    engine.on_model("done", "", "بس")
    assert engine.result() == {
        "profile": "clinic_appointment_confirmation",
        "outcome": "completed",
        "flags": ["attendance_known", "identity_ok"],
        "fields": {"preferred_time": "جمعرات شام"},
    }


def test_knowledge_retrieval_is_grounded_in_profile_files() -> None:
    _, documents = load_profile(CLINIC)
    kb = KnowledgeBase(documents)

    assert "parking" in kb.search("is there parking?")[0].lower()
    assert kb.search("weather in Lahore") == []


def test_profile_validation_rejects_broken_actions() -> None:
    profile, _ = load_profile(NRSP_PROFILE)
    data = profile.model_dump()
    data["actions"] = [{**data["actions"][0], "name": "reply"}, *data["actions"][1:]]
    with pytest.raises(ValidationError):
        AgentProfile.model_validate(data)


def test_missing_public_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="public fields"):
        build_engine(CLINIC, {"patient_name": "سارہ"})


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("چار پانچ چھ سات", "4567"),
        ("۴۵۶۷", "4567"),
        ("فور فائیو سکس سیون", "4567"),
        ("four five six seven", "4567"),
        ("ہندس سے 4، 5، 6، 7", "4567"),
    ],
)
def test_spoken_number_parser(text: str, digits: str) -> None:
    assert extract_digits(text) == digits

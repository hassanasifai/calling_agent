"""Deterministic call policy: identity verification, disclosure and exits.

The LLM never decides whether a caller is verified and never sees or speaks
customer facts. Every statement containing a name, amount or date comes from
the fixed templates in :mod:`nrsp_agent.conversation.script`, and only after
code-level verification.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from nrsp_agent.conversation import script


class CustomerContext(BaseModel):
    """Customer data for one call; spoken values are pre-rendered Urdu text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_name: str = Field(min_length=1, max_length=80)
    cnic_last4: str = Field(pattern=r"^\d{4}$")
    installment_amount_pkr: int = Field(gt=0)
    installment_amount_spoken: str = Field(min_length=1, max_length=120)
    due_date_spoken: str = Field(min_length=1, max_length=80)


class Stage(StrEnum):
    CONFIRM_IDENTITY = "confirm_identity"
    VERIFY = "verify"
    DISCUSS = "discuss"
    CLOSED = "closed"


class Outcome(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WRONG_PERSON = "wrong_person"
    VERIFICATION_FAILED = "verification_failed"
    OPTED_OUT = "opted_out"
    HANDOFF_REQUESTED = "handoff_requested"
    NO_RESPONSE = "no_response"


class Intent(StrEnum):
    OPT_OUT = "opt_out"
    HANDOFF = "handoff"
    GOODBYE = "goodbye"
    WHO_IS_CALLING = "who_is_calling"
    NEGATIVE = "negative"
    AFFIRMATIVE = "affirmative"
    ASK_AMOUNT = "ask_amount"
    ASK_DATE = "ask_date"
    PAYMENT_DIFFICULTY = "payment_difficulty"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    """What to do with one caller turn.

    Exactly one of ``speak`` (a fixed template) or ``use_llm`` (let the guarded
    LLM acknowledge the turn) is set.
    """

    speak: str | None = None
    use_llm: bool = False
    end_call: bool = False
    # Switch the ASR prompt for the caller's next answer, if set.
    asr_mode: AsrMode | None = None


class AsrMode(StrEnum):
    DOMAIN = "domain"
    DIGITS = "digits"


_ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍‎‏ًٌٍ" "َُِّْٰ"), None)
_LETTER_FOLD = str.maketrans(
    {"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ہ", "ة": "ہ", "أ": "ا", "إ": "ا"}
)
_DIGIT_FOLD = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Order matters: earlier intents win when several phrases appear.
_INTENT_PHRASES: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.OPT_OUT,
        (
            "کال نہ کریں",
            "کال مت کریں",
            "فون نہ کریں",
            "فون مت کریں",
            "رابطہ نہ کریں",
            "تنگ نہ کریں",
            "دوبارہ کال نہ",
            "کال نہیں چاہیے",
        ),
    ),
    (
        Intent.HANDOFF,
        (
            "نمائندہ",
            "نمائندے",
            "انسان",
            "کسی بندے",
            "آپریٹر",
            "ایجنٹ",
            "مینیجر",
            "افسر سے",
        ),
    ),
    (
        Intent.GOODBYE,
        # حافت/حافز: measured Whisper spellings of حافظ over G.711.
        (
            "خدا حافظ",
            "اللہ حافظ",
            "خدا حافت",
            "اللہ حافت",
            "خدا حافز",
            "اللہ حافز",
            "فی امان اللہ",
            "الوداع",
            "bye",
        ),
    ),
    (Intent.WHO_IS_CALLING, ("کون بول", "کون ہیں", "کون ہو", "کہاں سے", "کس کی طرف")),
    (
        Intent.ASK_AMOUNT,
        ("کتنی رقم", "کتنے پیسے", "کتنی قسط", "رقم کتنی", "قسط کتنی", "کتنے روپے"),
    ),
    (Intent.ASK_DATE, ("کب تک", "کس تاریخ", "تاریخ کیا", "آخری تاریخ", "کب ادا")),
    (
        Intent.PAYMENT_DIFFICULTY,
        (
            "مہلت",
            "توسیع",
            "پیسے نہیں",
            "رقم نہیں",
            "ادا نہیں کر سکتا",
            "ادا نہیں کر سکتی",
            "ادائیگی نہیں کر سکتا",
            "ادائیگی نہیں کر سکتی",
            "نہیں دے سکتا",
            "نہیں دے سکتی",
            "مشکل ہے",
        ),
    ),
    (Intent.NEGATIVE, ("نہیں", "غلط نمبر", "رانگ نمبر", "کوئی اور", "وہ نہیں")),
    (
        Intent.AFFIRMATIVE,
        (
            "جی",
            "ہاں",
            "بالکل",
            "درست",
            "صحیح",
            "ٹھیک",
            "میں ہی",
            "بول رہا",
            "بول رہی",
            "بات کر رہا",
            "بات کر رہی",
            "yes",
        ),
    ),
)

_DIGIT_WORDS = {
    "صفر": "0",
    "زیرو": "0",
    "ایک": "1",
    "دو": "2",
    "تین": "3",
    "چار": "4",
    "پانچ": "5",
    "چھ": "6",
    "چھے": "6",
    "چھہ": "6",
    "سات": "7",
    "آٹھ": "8",
    "نو": "9",
    "نؤ": "9",
}


def normalize(text: str) -> str:
    """Fold Urdu script variants, diacritics, digits and punctuation."""
    folded = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    folded = folded.translate(_LETTER_FOLD).translate(_DIGIT_FOLD).lower()
    return " ".join(_PUNCT.sub(" ", folded).split())


def detect_intent(text: str) -> Intent:
    tokens = normalize(text).split()
    normalized = f" {' '.join(tokens)} "
    # A goodbye ends the call, so it only counts at the end of the utterance
    # (at most one trailing word): ASR was measured inserting "اللہ حافظ"
    # mid-sentence, and acting on that would hang up on the caller.
    closing = f" {' '.join(tokens[-3:])} "
    for intent, phrases in _INTENT_PHRASES:
        haystack = closing if intent == Intent.GOODBYE else normalized
        if any(f" {phrase} " in haystack for phrase in phrases):
            return intent
    return Intent.UNKNOWN


def extract_digits(text: str) -> str:
    """Return spoken or written digits in order (Urdu words or numerals)."""
    digits: list[str] = []
    for token in normalize(text).split():
        if token.isdigit():
            digits.append(token)
        elif token in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[token])
    return "".join(digits)


def is_degenerate(text: str) -> bool:
    """Whether a transcript is a Whisper repetition loop, not speech.

    Measured: "نمائندہ، نمائندہ، نمائندہ، نمائندہ" from unclear audio. Such
    output must never trigger an intent (here: a handoff that hangs up).
    """
    tokens = normalize(text).split()
    if len(tokens) < 3:
        return False
    word = max(set(tokens), key=tokens.count)
    count = tokens.count(word)
    # A repeated digit ("ایک ایک ایک ایک") is a legitimate CNIC answer.
    return count >= 3 and count / len(tokens) >= 0.5 and not extract_digits(word)


MAX_SPOKEN_DIGITS = 6


def digits_match(heard: str, expected: str) -> bool:
    """Whether the expected digits were spoken, tolerating ASR padding.

    Whisper was measured adding a digit from the question ("آخری چار" → 4)
    or repeating the last one ("... سات، پانچ"). Accept the expected digits
    contiguously anywhere in at most ``MAX_SPOKEN_DIGITS`` heard digits; the
    cap stops a caller reciting 0-9 to cover every four-digit window. Each
    attempt therefore tests at most three windows.
    """
    return len(heard) <= MAX_SPOKEN_DIGITS and expected in heard


class ConversationPolicy:
    """One call's verification state machine."""

    def __init__(
        self,
        customer: CustomerContext,
        *,
        max_verify_attempts: int = 2,
        max_unclear_turns: int = 2,
        max_silence_prompts: int = 2,
        max_llm_turns: int = 3,
    ) -> None:
        self._customer = customer
        self._max_verify_attempts = max_verify_attempts
        self._max_unclear_turns = max_unclear_turns
        self._max_silence_prompts = max_silence_prompts
        self._max_llm_turns = max_llm_turns
        self.stage = Stage.CONFIRM_IDENTITY
        self.outcome = Outcome.IN_PROGRESS
        self.verified = False
        self.flags: set[str] = set()
        self._verify_failures = 0
        self._unclear_turns = 0
        self._silence_prompts = 0
        self._llm_turns = 0
        self._last_question = script.confirm_identity(customer.customer_name)

    def greeting(self) -> str:
        return script.greeting(self._customer.customer_name)

    def on_user_turn(self, text: str) -> Decision | None:
        """Decide the reply; ``None`` means ignore (call already closing)."""
        if self.stage == Stage.CLOSED:
            return None
        self._silence_prompts = 0
        if is_degenerate(text):
            return self._unclear()
        intent = detect_intent(text)
        if intent == Intent.OPT_OUT:
            return self._close(Outcome.OPTED_OUT, script.OPT_OUT)
        if intent == Intent.HANDOFF:
            return self._close(Outcome.HANDOFF_REQUESTED, script.HANDOFF)
        if self.stage == Stage.CONFIRM_IDENTITY:
            return self._confirm_identity(text, intent)
        if self.stage == Stage.VERIFY:
            return self._verify(text, intent)
        return self._discuss(intent)

    def on_silence(self) -> Decision | None:
        if self.stage == Stage.CLOSED:
            return None
        if self.stage == Stage.DISCUSS and self._last_question == script.ANYTHING_ELSE:
            # The call's purpose is done; silence after "anything else?" ends it.
            return self._close(Outcome.COMPLETED, script.CLOSING)
        self._silence_prompts += 1
        if self._silence_prompts > self._max_silence_prompts:
            return self._close(Outcome.NO_RESPONSE, script.NO_RESPONSE)
        return Decision(speak=f"{script.SILENCE_PROMPT} {self._last_question}")

    def _ask(
        self, text: str, question: str, asr_mode: AsrMode | None = None
    ) -> Decision:
        self._last_question = question
        return Decision(speak=f"{text} {question}".strip(), asr_mode=asr_mode)

    def _unclear(self) -> Decision:
        self._unclear_turns += 1
        if self._unclear_turns > self._max_unclear_turns:
            return self._close(Outcome.NO_RESPONSE, script.UNCLEAR_CLOSE)
        return self._ask(script.UNCLEAR, self._last_question)

    def _confirm_identity(self, text: str, intent: Intent) -> Decision:
        name_spoken = normalize(self._customer.customer_name) in normalize(text)
        if intent == Intent.WHO_IS_CALLING:
            return self._ask(script.WHO_WE_ARE, self._last_question)
        if intent == Intent.NEGATIVE and not name_spoken:
            return self._close(Outcome.WRONG_PERSON, script.WRONG_PERSON)
        if intent == Intent.AFFIRMATIVE or name_spoken:
            self.stage = Stage.VERIFY
            self._unclear_turns = 0
            return self._ask(script.THANKS, script.ASK_CNIC_LAST4, AsrMode.DIGITS)
        return self._unclear()

    def _verify(self, text: str, intent: Intent) -> Decision:
        digits = extract_digits(text)
        if len(digits) < 4:
            if intent == Intent.WHO_IS_CALLING:
                return self._ask(script.WHO_WE_ARE, self._last_question)
            return self._unclear()
        if digits_match(digits, self._customer.cnic_last4):
            self.stage = Stage.DISCUSS
            self.verified = True
            self._unclear_turns = 0
            return self._ask(
                script.VERIFIED,
                script.installment_notice(
                    self._customer.installment_amount_spoken,
                    self._customer.due_date_spoken,
                ),
                AsrMode.DOMAIN,
            )
        self._verify_failures += 1
        if self._verify_failures >= self._max_verify_attempts:
            return self._close(Outcome.VERIFICATION_FAILED, script.VERIFICATION_FAILED)
        return self._ask(script.VERIFY_RETRY, script.ASK_CNIC_LAST4)

    def _discuss(self, intent: Intent) -> Decision:
        anything_else_pending = self._last_question == script.ANYTHING_ELSE
        if intent == Intent.GOODBYE:
            return self._close(Outcome.COMPLETED, script.CLOSING)
        if intent == Intent.NEGATIVE and anything_else_pending:
            return self._close(Outcome.COMPLETED, script.CLOSING)
        if intent in (Intent.PAYMENT_DIFFICULTY, Intent.NEGATIVE):
            # A bare "no" to "can you pay on time?" is a payment difficulty.
            # Never left to the LLM: it was measured promising extensions.
            self.flags.add("payment_difficulty")
            self._last_question = script.ANYTHING_ELSE
            return Decision(speak=script.PAYMENT_DIFFICULTY)
        if intent == Intent.ASK_AMOUNT or intent == Intent.ASK_DATE:
            return self._ask(
                "",
                script.installment_notice(
                    self._customer.installment_amount_spoken,
                    self._customer.due_date_spoken,
                ),
            )
        if self._llm_turns >= self._max_llm_turns:
            return self._close(Outcome.COMPLETED, script.CLOSING)
        self._llm_turns += 1
        # The instruction (and the guard's fallback) end every reply with it.
        self._last_question = script.ANYTHING_ELSE
        return Decision(use_llm=True)

    def _close(self, outcome: Outcome, text: str) -> Decision:
        self.stage = Stage.CLOSED
        self.outcome = outcome
        return Decision(speak=text, end_call=True)

"""Fixed Urdu call templates.

Every utterance that discloses identity, amounts or dates is defined here and
spoken verbatim; the LLM only produces short replies under
``DISCUSS_INSTRUCTION`` and its output is guarded. Brand terms are written in
Urdu script ("این آر ایس پی", "اے آئی") so the Urdu Piper voice pronounces them.
The Piper ``ur_PK-fasih`` voice is male, hence masculine first-person verbs.
"""

from __future__ import annotations


def greeting(customer_name: str) -> str:
    return (
        "السلام علیکم۔ میں این آر ایس پی کی طرف سے ایک خودکار اے آئی معاون "
        f"بات کر رہا ہوں۔ {confirm_identity(customer_name)}"
    )


def confirm_identity(customer_name: str) -> str:
    return f"کیا میری بات {customer_name} سے ہو رہی ہے؟"


def installment_notice(amount_spoken: str, due_date_spoken: str) -> str:
    return (
        f"آپ کی {amount_spoken} روپے کی قسط {due_date_spoken} کو واجب الادا ہے۔ "
        "کیا آپ مقررہ تاریخ تک ادائیگی کر سکیں گے؟"
    )


WHO_WE_ARE = (
    "میں این آر ایس پی کا خودکار اے آئی معاون ہوں اور آپ کے قرض کے سلسلے میں "
    "کال کر رہا ہوں۔"
)
THANKS = "شکریہ۔"
ASK_CNIC_LAST4 = "تصدیق کے لیے براہِ کرم اپنے شناختی کارڈ نمبر کے آخری چار ہندسے بتائیں۔"
VERIFIED = "شکریہ، آپ کی تصدیق ہو گئی ہے۔"
VERIFY_RETRY = "معذرت، یہ ہندسے ہمارے ریکارڈ سے مطابقت نہیں رکھتے۔"
VERIFICATION_FAILED = (
    "معذرت، تصدیق مکمل نہیں ہو سکی۔ ہمارا نمائندہ آپ سے جلد رابطہ کرے گا۔ " "اللہ حافظ۔"
)
WRONG_PERSON = "زحمت کے لیے معذرت۔ آپ کا شکریہ، اللہ حافظ۔"
OPT_OUT = "جی بالکل، آپ کی درخواست نوٹ کر لی گئی ہے۔ زحمت کے لیے معذرت، اللہ حافظ۔"
HANDOFF = "جی ضرور۔ ہمارا نمائندہ جلد آپ سے رابطہ کرے گا۔ شکریہ، اللہ حافظ۔"
CLOSING = "آپ کے وقت کا بہت شکریہ۔ اللہ حافظ۔"
NO_RESPONSE = (
    "لگتا ہے آپ کی آواز نہیں آ رہی۔ ہم آپ سے بعد میں دوبارہ رابطہ کریں گے۔ "
    "اللہ حافظ۔"
)
UNCLEAR = "معذرت، میں آپ کی بات ٹھیک سے سمجھ نہیں سکا۔"
UNCLEAR_CLOSE = (
    "معذرت، رابطے میں مشکل ہو رہی ہے۔ ہم آپ سے بعد میں دوبارہ رابطہ کریں گے۔ "
    "اللہ حافظ۔"
)
SILENCE_PROMPT = "کیا آپ میری آواز سن رہے ہیں؟"
LLM_FALLBACK = "جی، آپ کا شکریہ۔ کیا آپ کچھ اور پوچھنا چاہیں گے؟"

# Whisper initial prompts. Domain nouns only: a prompt with "اللہ حافظ" was
# measured making Whisper hear a goodbye mid-sentence; this one scored 9/9 on
# the caller fixtures (previous prompt 6/9, none 7/9). The digit prompt is used
# only while digits are requested: over G.711 it turned "چارڈ، ماد، چیساہر"
# into "چار، پانچ، چھ، سات", but elsewhere it makes Whisper invent digits. It
# lists generic digit words and must never contain the customer's own digits.
# Contains no word that triggers an intent: on unclear audio Whisper was also
# measured looping the prompt's last word ("نمائندہ، نمائندہ، ..."), which read
# as a handoff request and hung up. Missing a handoff costs a repeat; a false
# one ends the call.
ASR_PROMPT_DOMAIN = "این آر ایس پی، قرض، قسط، ادائیگی، شناختی کارڈ، ہندسے"
ASR_PROMPT_DIGITS = (
    "شناختی کارڈ کے آخری چار ہندسے: ایک، دو، تین، چار، پانچ، چھ، سات، آٹھ، نو، صفر"
)

ANYTHING_ELSE = "کیا آپ کچھ اور پوچھنا چاہیں گے؟"
PAYMENT_DIFFICULTY = (
    "میں آپ کی بات نوٹ کر رہا ہوں۔ ادائیگی کے انتظام کے بارے میں ہمارا نمائندہ "
    f"آپ سے رابطہ کرے گا۔ {ANYTHING_ELSE}"
)

# Measured with qwen3:1.7b: a short English instruction plus one example and
# only the caller's last turn (~280 prompt tokens) gave clean Urdu replies at
# 0.25-0.5 s first token, while an Urdu instruction with the full history
# (~680 tokens) repeated amounts and invented facilities. Output is still
# guarded, and payment-difficulty turns never reach the LLM.
DISCUSS_INSTRUCTION = (
    "You are the automated AI phone assistant of NRSP, a microfinance organisation, "
    "on a call with a borrower whose identity is already verified. You just reminded "
    "them that an installment is due and asked whether they can pay on time.\n"
    "Reply to the borrower's last message in natural, polite Urdu (Urdu script only), "
    "in one or two short sentences.\n"
    "Rules: never mention any amount, number, digit or date; never promise discounts, "
    "extensions or waivers; never ask for passwords, PINs, OTPs or full CNIC numbers; "
    "do not repeat the borrower's words back; if you do not know the answer, say that "
    "an NRSP representative will contact them. "
    "End by asking if they need anything else."
)
DISCUSS_EXAMPLE = (
    {"role": "user", "content": "ٹھیک ہے، میں کوشش کروں گا۔"},
    {
        "role": "assistant",
        "content": "بہت شکریہ، ہم آپ کے تعاون کے شکر گزار ہیں۔ "
        "کیا آپ کچھ اور پوچھنا چاہیں گے؟",
    },
)

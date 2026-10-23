"""Urdu text normalisation and spoken-number parsing (language utilities)."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍‎‏ًٌٍ" "َُِّْٰ"), None)
_LETTER_FOLD = str.maketrans(
    {"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ہ", "ة": "ہ", "أ": "ا", "إ": "ا"}
)
_DIGIT_FOLD = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

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
    # English digit names: Pakistani callers often read CNIC digits in
    # English; Whisper writes them in Urdu script or Latin.
    "ون": "1",
    "ٹو": "2",
    "تھری": "3",
    "فور": "4",
    "فائیو": "5",
    "فائو": "5",
    "سکس": "6",
    "سیون": "7",
    "ایٹ": "8",
    "نائن": "9",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    # Roman Urdu digit names (callers and code-switched ASR output). "no"
    # and "do" are left out: they are also common English words.
    "sifar": "0",
    "ek": "1",
    "teen": "3",
    "tin": "3",
    "char": "4",
    "chaar": "4",
    "panch": "5",
    "paanch": "5",
    "chay": "6",
    "chhe": "6",
    "chhay": "6",
    "che": "6",
    "saat": "7",
    "sat": "7",
    "aath": "8",
    "ath": "8",
    "nau": "9",
    "nao": "9",
}


DIGIT_NAMES = ("صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو")


def normalize(text: str) -> str:
    """Fold Urdu script variants, diacritics, digits and punctuation."""
    folded = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    folded = folded.translate(_LETTER_FOLD).translate(_DIGIT_FOLD).lower()
    return " ".join(_PUNCT.sub(" ", folded).split())


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


def spoken_digits(digits: str) -> str:
    """Urdu words for a digit string, for reading numbers back."""
    return "، ".join(DIGIT_NAMES[int(d)] for d in digits if d.isdigit())

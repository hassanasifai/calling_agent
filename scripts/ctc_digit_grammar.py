"""Grammar-constrained spoken-digit decoding over CTC frame log-probabilities.

Free decoding of short isolated digit words over a phone line produces
near-miss spellings ("سال" for "سات"). When the agent has asked for digits,
the acoustic model's frame posteriors are instead decoded against a small
grammar: filler* (digit-word filler*)+, where filler absorbs any other words
or pauses at a fixed per-frame penalty. It is unbiased: the grammar contains
every digit equally and never the expected answer, so it cannot make a wrong
answer match; the engine still reads the digits back for confirmation.
"""

from __future__ import annotations

import numpy as np

# Spellings per digit: Urdu, and English digit names as written in Urdu.
DIGIT_WORDS: dict[str, tuple[str, ...]] = {
    "0": ("صفر", "زیرو"),
    "1": ("ایک", "ون"),
    "2": ("دو", "ٹو"),
    "3": ("تین", "تھری"),
    "4": ("چار", "فور"),
    "5": ("پانچ", "فائیو", "فائو"),
    "6": ("چھ", "چھے", "سکس"),
    "7": ("سات", "سیون"),
    "8": ("آٹھ", "ایٹ"),
    "9": ("نو", "نائن"),
}


class DigitGrammar:
    """Viterbi decoder for a digit-word loop with a filler model."""

    def __init__(self, tokens: list[str], blank: int = 0, filler_penalty=2.5):
        index = {t: i for i, t in enumerate(tokens)}
        self.blank = blank
        self.filler_penalty = filler_penalty
        self.words: list[tuple[str, list[int]]] = []
        for digit, spellings in DIGIT_WORDS.items():
            for spelling in spellings:
                ids = [index[ch] for ch in spelling if ch in index]
                if len(ids) == len(spelling):
                    self.words.append((digit, ids))

    def decode(self, log_probs: np.ndarray) -> str:
        """Most likely digit string for ``log_probs`` of shape (T, V)."""
        lp = np.asarray(log_probs, dtype=np.float64)
        nonblank = np.delete(lp, self.blank, axis=1).max(axis=1)
        filler = np.maximum(lp[:, self.blank], nonblank - self.filler_penalty)
        neg = -np.inf
        # Per word: states 2j = char j, 2j+1 = blank after char j.
        scores = [np.full(2 * len(ids), neg) for _, ids in self.words]
        paths: list[list[str]] = [[""] * (2 * len(ids)) for _, ids in self.words]
        g_score, g_path = 0.0, ""
        for t in range(lp.shape[0]):
            frame = lp[t]
            new_g, new_g_path = g_score + filler[t], g_path
            new_scores, new_paths = [], []
            for w, (digit, ids) in enumerate(self.words):
                old, old_paths = scores[w], paths[w]
                n = len(ids)
                cur = np.full(2 * n, neg)
                cur_paths = [""] * (2 * n)
                for j in range(n):
                    # Char state j: stay, from blank j-1, from char j-1 (if
                    # different char), or enter the word from filler (j == 0).
                    cands = [(old[2 * j], old_paths[2 * j])]
                    if j == 0:
                        cands.append((g_score, g_path))
                    else:
                        cands.append((old[2 * j - 1], old_paths[2 * j - 1]))
                        if ids[j] != ids[j - 1]:
                            cands.append((old[2 * j - 2], old_paths[2 * j - 2]))
                    best, path = max(cands, key=lambda c: c[0])
                    cur[2 * j], cur_paths[2 * j] = best + frame[ids[j]], path
                    # Blank after char j.
                    stay, came = old[2 * j + 1], old[2 * j]
                    if stay >= came:
                        cur[2 * j + 1] = stay + frame[self.blank]
                        cur_paths[2 * j + 1] = old_paths[2 * j + 1]
                    else:
                        cur[2 * j + 1] = came + frame[self.blank]
                        cur_paths[2 * j + 1] = old_paths[2 * j]
                # Leaving the word (after its last char or trailing blank)
                # emits the digit and returns to filler for the next frame.
                end = max((cur[2 * n - 2], 2 * n - 2), (cur[2 * n - 1], 2 * n - 1))
                if end[0] > new_g:
                    new_g, new_g_path = end[0], cur_paths[end[1]] + digit
                new_scores.append(cur)
                new_paths.append(cur_paths)
            scores, paths = new_scores, new_paths
            g_score, g_path = new_g, new_g_path
        return g_path

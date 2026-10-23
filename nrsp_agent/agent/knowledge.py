"""Small in-memory BM25 retrieval over a profile's knowledge files.

Knowledge bases for a call are a few pages; a dependency-free BM25 over
paragraph chunks retrieves in well under a millisecond and keeps answers
grounded without vector stores or extra services.
"""

from __future__ import annotations

import math
from collections import Counter

from nrsp_agent.agent.textnorm import normalize


def _chunks(text: str) -> list[str]:
    parts, heading = [], ""
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#") and "\n" not in block:
            heading = block.lstrip("# ")
            continue
        parts.append(f"{heading}: {block}" if heading else block)
    return parts


class KnowledgeBase:
    def __init__(self, documents: list[tuple[str, str]], k1: float = 1.4, b=0.75):
        self.chunks = [c for _, text in documents for c in _chunks(text)]
        self._docs = [normalize(c).split() for c in self.chunks]
        self._tf = [Counter(d) for d in self._docs]
        n = len(self._docs) or 1
        df = Counter(t for d in self._docs for t in set(d))
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg = sum(map(len, self._docs)) / n if self._docs else 0
        self._k1, self._b = k1, b

    def search(self, query: str, top_k: int = 2, min_score: float = 1.5) -> list[str]:
        terms = [t for t in normalize(query).split() if t in self._idf]
        if not terms:
            return []
        scored = []
        for i, tf in enumerate(self._tf):
            length = len(self._docs[i])
            score = sum(
                self._idf[t]
                * tf[t]
                * (self._k1 + 1)
                / (tf[t] + self._k1 * (1 - self._b + self._b * length / self._avg))
                for t in terms
                if t in tf
            )
            if score >= min_score:
                scored.append((score, i))
        return [self.chunks[i] for _, i in sorted(scored, reverse=True)[:top_k]]

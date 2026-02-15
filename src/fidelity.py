from __future__ import annotations

from typing import Protocol

from tokenizer import Tokenizer, WhitespaceTokenizer


class FidelityVerifier(Protocol):
    def score(self, source_text: str, rewritten_text: str) -> float:
        ...


class NoOpVerifier:
    def score(self, source_text: str, rewritten_text: str) -> float:
        return 1.0


class TokenJaccardVerifier:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or WhitespaceTokenizer()

    def score(self, source_text: str, rewritten_text: str) -> float:
        source_tokens = set(self._tokenizer.encode(source_text))
        rewritten_tokens = set(self._tokenizer.encode(rewritten_text))
        if not source_tokens and not rewritten_tokens:
            return 1.0
        if not source_tokens or not rewritten_tokens:
            return 0.0
        intersection = len(source_tokens & rewritten_tokens)
        union = len(source_tokens | rewritten_tokens)
        if union == 0:
            return 0.0
        return intersection / union

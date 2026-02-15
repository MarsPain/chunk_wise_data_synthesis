from __future__ import annotations

import re
from typing import List, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[str]:
        ...

    def decode(self, tokens: List[str]) -> str:
        ...


class WhitespaceTokenizer:
    _token_pattern = re.compile(r"\S+")

    def encode(self, text: str) -> List[str]:
        return self._token_pattern.findall(text)

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


def take_last_tokens(text: str, tokenizer: Tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return tokenizer.decode(tokens[-max_tokens:])

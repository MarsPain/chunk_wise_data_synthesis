from __future__ import annotations

"""Legacy tokenizer module kept for backward compatibility."""

from tokenization import Tokenizer, WhitespaceTokenizer, take_last_tokens

__all__ = [
    "Tokenizer",
    "WhitespaceTokenizer",
    "take_last_tokens",
]

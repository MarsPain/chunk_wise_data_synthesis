from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _words_in_order(words: list[str], text: str) -> bool:
    if not words:
        return True
    text_words = text.split()
    word_idx = 0

    for text_word in text_words:
        text_word_clean = text_word.strip(".,;:!?()[]{}\"'")
        if text_word_clean == words[word_idx]:
            word_idx += 1
            if word_idx >= len(words):
                return True

    return False


__all__ = [
    "_tokenize",
    "_token_jaccard",
    "_words_in_order",
]

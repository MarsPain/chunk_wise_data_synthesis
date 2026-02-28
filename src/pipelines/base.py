from __future__ import annotations

from typing import List

from tokenization import Tokenizer


def _longest_overlap(
    left_tokens: List[str],
    right_tokens: List[str],
    max_overlap_tokens: int,
) -> int:
    max_size = min(len(left_tokens), len(right_tokens), max_overlap_tokens)
    for size in range(max_size, 0, -1):
        if left_tokens[-size:] == right_tokens[:size]:
            return size
    return 0


def stitch_rewritten_chunks(
    chunks: List[str],
    tokenizer: Tokenizer,
    max_overlap_tokens: int = 96,
) -> str:
    if not chunks:
        return ""

    merged_tokens = tokenizer.encode(chunks[0])
    for chunk in chunks[1:]:
        right_tokens = tokenizer.encode(chunk)
        overlap = _longest_overlap(merged_tokens, right_tokens, max_overlap_tokens)
        merged_tokens.extend(right_tokens[overlap:])
    return tokenizer.decode(merged_tokens).strip()


__all__ = [
    "_longest_overlap",
    "stitch_rewritten_chunks",
]

from __future__ import annotations

import re
from typing import List

from tokenizer import Tokenizer


def split_into_structural_units(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    units = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    return units


def split_into_token_chunks(
    text: str,
    tokenizer: Tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    stride = chunk_tokens - overlap_tokens
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        window = tokens[start:end]
        if not window:
            break
        chunks.append(tokenizer.decode(window))
        if end >= len(tokens):
            break
        start += stride
    return chunks


def split_document_into_chunks(
    text: str,
    tokenizer: Tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    units = split_into_structural_units(text)
    if not units:
        return []

    chunks: List[str] = []
    current_tokens: List[str] = []

    def flush_current() -> None:
        nonlocal current_tokens
        if current_tokens:
            chunks.append(tokenizer.decode(current_tokens))
            current_tokens = []

    for unit in units:
        unit_tokens = tokenizer.encode(unit)
        if len(unit_tokens) > chunk_tokens:
            flush_current()
            chunks.extend(
                split_into_token_chunks(
                    text=unit,
                    tokenizer=tokenizer,
                    chunk_tokens=chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue

        if len(current_tokens) + len(unit_tokens) <= chunk_tokens:
            current_tokens.extend(unit_tokens)
            continue

        flush_current()
        current_tokens.extend(unit_tokens)

    flush_current()
    return chunks

from __future__ import annotations

import re
from typing import List, Literal

from tokenization import Tokenizer


def detect_dominant_script(text: str) -> Literal["latin", "cjk", "mixed"]:
    """
    检测文本主导文字类型
    - latin: 拉丁字母（英文等）
    - cjk: 中日韩文字
    - mixed: 混合
    """
    # CJK Unicode 范围
    cjk_pattern = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
    )
    latin_pattern = re.compile(r"[a-zA-Z]")
    
    cjk_count = len(cjk_pattern.findall(text))
    latin_count = len(latin_pattern.findall(text))
    total_chars = len(re.sub(r"\s", "", text))
    
    if total_chars == 0:
        return "latin"
    
    cjk_ratio = cjk_count / total_chars
    latin_ratio = latin_count / total_chars
    
    if cjk_ratio > 0.5:
        return "cjk"
    elif latin_ratio > 0.5:
        return "latin"
    else:
        return "mixed"


def get_adaptive_length(
    text: str,
    tokenizer: Tokenizer,
    mode: Literal["auto", "token", "char"] = "auto",
) -> int:
    """
    自适应获取文本长度
    - auto: 根据文字类型自动选择
    - token: 强制使用 token 数
    - char: 强制使用字符数
    """
    if mode == "char":
        return len(text)
    if mode == "token":
        return len(tokenizer.encode(text))
    
    # auto 模式
    script = detect_dominant_script(text)
    if script == "latin":
        return len(tokenizer.encode(text))
    else:  # cjk 或 mixed，使用字符数更稳定
        return len(text)


def split_into_structural_units(text: str) -> List[str]:
    """按空行（段落）分割"""
    stripped = text.strip()
    if not stripped:
        return []
    units = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    return units


def split_into_lines(text: str) -> List[str]:
    """按单换行分割，保留非空行"""
    lines = [line.strip() for line in text.split("\n")]
    return [line for line in lines if line]


def split_into_char_chunks(
    text: str,
    max_chars: int,
) -> List[str]:
    """按字符分割（无 overlap）"""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    
    chunks: List[str] = []
    for i in range(0, len(text), max_chars):
        chunk = text[i : i + max_chars]
        if chunk:
            chunks.append(chunk)
    return chunks


def split_into_token_chunks_no_overlap(
    text: str,
    tokenizer: Tokenizer,
    chunk_tokens: int,
) -> List[str]:
    """按 token 分割（无 overlap）"""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    
    tokens = tokenizer.encode(text)
    chunks: List[str] = []
    
    for i in range(0, len(tokens), chunk_tokens):
        window = tokens[i : i + chunk_tokens]
        if window:
            chunks.append(tokenizer.decode(window))
    
    return chunks


def split_document_into_chunks(
    text: str,
    tokenizer: Tokenizer,
    chunk_size: int,
    length_mode: Literal["auto", "token", "char"] = "auto",
    enable_line_fallback: bool = True,
    enable_char_fallback: bool = True,
) -> List[str]:
    """
    分层降级分割策略：
    1. 优先按空行（段落）分割
    2. 段落超长则按单行分割（可选）
    3. 单行仍超长则按字符分割（可选，token模式时按token分割）
    
    Args:
        text: 输入文本
        tokenizer: 分词器
        chunk_size: 长度限制（token数或字符数，根据 mode）
        length_mode: 长度计算模式
        enable_line_fallback: 超长时是否启用单行分割
        enable_char_fallback: 单行超长时是否启用字符分割
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def get_length(t: str) -> int:
        return get_adaptive_length(t, tokenizer, length_mode)
    
    # 判断实际使用的长度模式（处理 auto 的情况）
    def effective_mode(t: str) -> Literal["token", "char"]:
        if length_mode == "auto":
            script = detect_dominant_script(t)
            return "token" if script == "latin" else "char"
        return length_mode  # type: ignore

    chunks: List[str] = []
    current_buffer: List[str] = []
    current_len = 0

    def flush_buffer() -> None:
        nonlocal current_buffer, current_len
        if current_buffer:
            chunks.append(" ".join(current_buffer))
            current_buffer = []
            current_len = 0

    def add_to_buffer(item: str, item_len: int) -> None:
        nonlocal current_buffer, current_len
        current_buffer.append(item)
        current_len += item_len
    
    def split_by_tokens(t: str) -> List[str]:
        """使用 token 级分割"""
        return split_into_token_chunks_no_overlap(t, tokenizer, chunk_size)

    # Level 1: 按段落分割
    paragraphs = split_into_structural_units(text)
    if not paragraphs:
        return []

    for para in paragraphs:
        para_len = get_length(para)

        # 情况A: 段落在限制内，尝试累积
        if para_len <= chunk_size:
            if current_len + para_len <= chunk_size:
                add_to_buffer(para, para_len)
            else:
                flush_buffer()
                add_to_buffer(para, para_len)
            continue

        # 情况B: 段落超长，需要降级处理
        flush_buffer()  # 先 flush 已累积的内容

        if enable_line_fallback:
            # Level 2: 按单行分割
            lines = split_into_lines(para)
            for line in lines:
                line_len = get_length(line)

                if line_len <= chunk_size:
                    # 单行在限制内，尝试累积
                    if current_len + line_len <= chunk_size:
                        add_to_buffer(line, line_len)
                    else:
                        flush_buffer()
                        add_to_buffer(line, line_len)
                else:
                    # 单行还超长
                    flush_buffer()

                    if enable_char_fallback:
                        # Level 3: 按 token 或字符分割
                        if effective_mode(line) == "token":
                            token_chunks = split_by_tokens(line)
                            chunks.extend(token_chunks)
                        else:
                            char_chunks = split_into_char_chunks(line, chunk_size)
                            chunks.extend(char_chunks)
                    else:
                        # 不启用字符分割，直接截断或保留原样
                        chunks.append(line)
        else:
            # 不启用行分割，直接按 token 或字符分割
            if enable_char_fallback:
                if effective_mode(para) == "token":
                    token_chunks = split_by_tokens(para)
                    chunks.extend(token_chunks)
                else:
                    char_chunks = split_into_char_chunks(para, chunk_size)
                    chunks.extend(char_chunks)
            else:
                chunks.append(para)

    flush_buffer()
    return chunks

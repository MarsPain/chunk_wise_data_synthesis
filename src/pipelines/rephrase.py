from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal

from chunking import split_document_into_chunks
from quality.fidelity import FidelityVerifier, NoOpVerifier
from model import RewriteModel
from pipelines.base import stitch_rewritten_chunks
from prompts.rephrase import RewriteRequest
from tokenization import Tokenizer, take_last_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    chunk_size: int = 1024  # 长度限制（token数或字符数，根据 length_mode）
    length_mode: Literal["auto", "token", "char"] = "auto"  # 长度计算模式
    enable_line_fallback: bool = True  # 超长时启用单行分割
    enable_char_fallback: bool = True  # 单行超长时启用字符分割
    prefix_window_tokens: int = 1024
    fidelity_threshold: float = 0.85
    max_retries: int = 2
    anchor_tokens: int = 256
    max_stitch_overlap_tokens: int = 96
    global_anchor_mode: Literal["none", "head"] = "head"
    default_style_instruction: str = "neutral factual rewrite"
    prompt_language: Literal["en", "zh"] = "en"

    # 兼容旧参数名
    @property
    def chunk_tokens(self) -> int:
        return self.chunk_size

    @property
    def overlap_tokens(self) -> int:
        return 0  # 不再使用 token 级 overlap


class ChunkWiseRephrasePipeline:
    def __init__(
        self,
        model: RewriteModel,
        tokenizer: Tokenizer,
        config: PipelineConfig | None = None,
        verifier: FidelityVerifier | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or PipelineConfig()
        self._verifier = verifier or NoOpVerifier()

    def run(self, text: str, style_instruction: str = "") -> str:
        if not text.strip():
            return ""

        chunks = split_document_into_chunks(
            text=text,
            tokenizer=self._tokenizer,
            chunk_size=self._config.chunk_size,
            length_mode=self._config.length_mode,
            enable_line_fallback=self._config.enable_line_fallback,
            enable_char_fallback=self._config.enable_char_fallback,
        )
        logger.info(f"Document split into {len(chunks)} chunk(s)")
        for i, chk in enumerate(chunks):
            chk_tokens = len(self._tokenizer.encode(chk))
            logger.debug(f"  Chunk {i+1}: {len(chk)} chars, ~{chk_tokens} tokens")

        global_anchor = self._build_global_anchor(text)
        if global_anchor:
            logger.info(f"Global anchor built: {len(global_anchor)} chars")
        rewritten_chunks: List[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_num = idx + 1
            logger.info(f"Processing chunk {chunk_num}/{len(chunks)}...")
            generated_prefix = self._build_generated_prefix(rewritten_chunks)
            if generated_prefix:
                prefix_tokens = len(self._tokenizer.encode(generated_prefix))
                logger.debug(
                    f"  Generated prefix: {len(generated_prefix)} chars, ~{prefix_tokens} tokens"
                )
            rewritten = self._rewrite_chunk_with_retries(
                chunk=chunk,
                generated_prefix=generated_prefix,
                global_anchor=global_anchor,
                style_instruction=style_instruction,
                chunk_num=chunk_num,
                total_chunks=len(chunks),
            )
            rewritten_chunks.append(rewritten)
            logger.info(f"  Chunk {chunk_num} done: {len(chunk)} -> {len(rewritten)} chars")

        return stitch_rewritten_chunks(
            chunks=rewritten_chunks,
            tokenizer=self._tokenizer,
            max_overlap_tokens=self._config.max_stitch_overlap_tokens,
        )

    def _build_generated_prefix(self, rewritten_chunks: List[str]) -> str:
        if not rewritten_chunks:
            return ""
        combined = " ".join(rewritten_chunks)
        return take_last_tokens(
            text=combined,
            tokenizer=self._tokenizer,
            max_tokens=self._config.prefix_window_tokens,
        )

    def _build_global_anchor(self, text: str) -> str:
        if self._config.global_anchor_mode == "none":
            return ""
        tokens = self._tokenizer.encode(text)
        return self._tokenizer.decode(tokens[: self._config.anchor_tokens])

    def _rewrite_chunk_with_retries(
        self,
        chunk: str,
        generated_prefix: str,
        global_anchor: str,
        style_instruction: str,
        chunk_num: int = 0,
        total_chunks: int = 0,
    ) -> str:
        best_candidate = ""
        best_score = -1.0
        best_issues: list[str] = []
        instruction = style_instruction or self._config.default_style_instruction
        retry_limit = max(self._config.max_retries, 1)
        chunk_tokens = len(self._tokenizer.encode(chunk))

        for retry_index in range(retry_limit):
            logger.debug(f"  Chunk {chunk_num}: retry {retry_index + 1}/{retry_limit}")
            request = RewriteRequest(
                style_instruction=instruction,
                global_anchor=global_anchor,
                generated_prefix=generated_prefix,
                current_chunk=chunk,
                retry_index=retry_index,
                strict_fidelity=retry_index > 0,
                prompt_language=self._config.prompt_language,
            )
            candidate = self._model.rewrite(request).strip()
            if not candidate:
                logger.warning(f"  Chunk {chunk_num}: model returned empty, using original")
                candidate = chunk

            score = self._verifier.score(chunk, candidate)
            issues = self._verifier.get_issues(chunk, candidate)
            logger.debug(f"  Chunk {chunk_num}: fidelity score = {score:.3f}")
            if issues:
                logger.debug(f"  Chunk {chunk_num}: fidelity issues = {issues}")

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_issues = issues
            if score >= self._config.fidelity_threshold:
                logger.info(f"  Chunk {chunk_num}: fidelity threshold met (score={score:.3f})")
                if issues:
                    for issue in issues:
                        logger.warning(f"  Chunk {chunk_num}: fidelity issue - {issue}")
                return candidate

        if retry_limit > 1:
            logger.info(f"  Chunk {chunk_num}: using best candidate (score={best_score:.3f})")
        if best_issues:
            for issue in best_issues:
                logger.warning(f"  Chunk {chunk_num}: fidelity issue - {issue}")
        return best_candidate or chunk


__all__ = [
    "PipelineConfig",
    "ChunkWiseRephrasePipeline",
]

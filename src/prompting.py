from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PromptLanguage = Literal["en", "zh"]


def _resolve_prompt_language(prompt_language: str) -> PromptLanguage:
    return "zh" if prompt_language == "zh" else "en"


@dataclass(frozen=True)
class RewriteRequest:
    style_instruction: str
    global_anchor: str
    generated_prefix: str
    current_chunk: str
    retry_index: int
    strict_fidelity: bool
    prompt_language: PromptLanguage = "en"


def render_rewrite_prompt(request: RewriteRequest) -> str:
    prompt_language = _resolve_prompt_language(request.prompt_language)
    style = request.style_instruction or "neutral rewrite"
    if prompt_language == "zh":
        strict_hint = (
            "长度尽量接近原文，不要摘要，保留所有事实。"
            if request.strict_fidelity
            else "在改写时保留事实和结构。"
        )
        return "\n\n".join(
            [
                "你是一名忠实改写助手。",
                f"风格：{style}",
                f"全局锚点：\n{request.global_anchor or '(无)'}",
                f"已生成前缀：\n{request.generated_prefix or '(无)'}",
                f"当前分块：\n{request.current_chunk}",
                f"约束：{strict_hint}",
                "只输出当前分块的改写结果。",
            ]
        )

    strict_hint = (
        "Keep length close to source, do not summarize, preserve all facts."
        if request.strict_fidelity
        else "Preserve facts and structure while rephrasing."
    )
    return "\n\n".join(
        [
            "You are a faithful rewriter.",
            f"Style: {style}",
            f"Global anchor:\n{request.global_anchor or '(none)'}",
            f"Generated prefix:\n{request.generated_prefix or '(none)'}",
            f"Current chunk:\n{request.current_chunk}",
            f"Constraint: {strict_hint}",
            "Output only the rewritten current chunk.",
        ]
    )

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteRequest:
    style_instruction: str
    global_anchor: str
    generated_prefix: str
    current_chunk: str
    retry_index: int
    strict_fidelity: bool


def render_rewrite_prompt(request: RewriteRequest) -> str:
    style = request.style_instruction or "neutral rewrite"
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

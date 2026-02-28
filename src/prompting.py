from __future__ import annotations

"""Legacy rephrase prompting module kept for backward compatibility."""

from prompts.base import PromptLanguage, _resolve_prompt_language
from prompts.rephrase import RewriteRequest, render_rewrite_prompt

__all__ = [
    "PromptLanguage",
    "_resolve_prompt_language",
    "RewriteRequest",
    "render_rewrite_prompt",
]

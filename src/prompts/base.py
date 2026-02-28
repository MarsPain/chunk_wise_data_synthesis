from __future__ import annotations

from typing import Literal


PromptLanguage = Literal["en", "zh"]


def _resolve_prompt_language(prompt_language: str) -> PromptLanguage:
    return "zh" if prompt_language == "zh" else "en"


def _none_text(prompt_language: PromptLanguage) -> str:
    return "(无)" if prompt_language == "zh" else "(none)"

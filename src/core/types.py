from __future__ import annotations

"""Unified type exports for public consumption.

This module provides a stable import surface and intentionally keeps type
ownership in domain modules (`prompting.py`, `model.py`, `generation_types.py`).
"""

from generation_types import (
    GenerationPlan,
    GenerationResult,
    GenerationState,
    QualityReport,
    SectionSpec,
)
from model import LLMRequest, LLMTask
from prompting import PromptLanguage, RewriteRequest

__all__ = [
    "LLMTask",
    "LLMRequest",
    "RewriteRequest",
    "PromptLanguage",
    "GenerationPlan",
    "SectionSpec",
    "GenerationState",
    "QualityReport",
    "GenerationResult",
]

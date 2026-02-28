from __future__ import annotations

"""Backward-compatible OpenAI backend import surface.

This module intentionally re-exports the backend implementation from
`backends.openai` to preserve existing imports.
"""

from backends.openai import (
    DEFAULT_BASE_URL,
    DEFAULT_BASE_URL_ENV_VAR,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ENV_VAR,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    OpenAIBackendConfig,
    OpenAILLMModel,
    OpenAIRewriteModel,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_BASE_URL_ENV_VAR",
    "DEFAULT_MODEL_ENV_VAR",
    "OpenAIBackendConfig",
    "OpenAILLMModel",
    "OpenAIRewriteModel",
]

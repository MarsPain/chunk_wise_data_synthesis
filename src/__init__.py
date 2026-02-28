from __future__ import annotations

"""Unified public API for the refactored package layout."""

from backends import OpenAIBackendConfig, OpenAILLMModel, OpenAIRewriteModel
from chunking import (
    detect_dominant_script,
    get_adaptive_length,
    split_document_into_chunks,
    split_into_char_chunks,
    split_into_lines,
    split_into_structural_units,
)
from core import config, protocols, types
from generation_state import initialize_state, update_state
from generation_types import (
    GenerationConfig,
    GenerationPlan,
    GenerationResult,
    GenerationState,
    QualityReport,
    SectionSpec,
)
from model import LLMModel, LLMRequest, LLMTask, RewriteModel, RewriteModelFromLLM
from pipelines import (
    ChunkWiseGenerationPipeline,
    ChunkWiseRephrasePipeline,
    PipelineConfig,
    _longest_overlap,
    stitch_rewritten_chunks,
)
from prompts import (
    PromptLanguage,
    RewriteRequest,
    render_consistency_prompt,
    render_plan_prompt,
    render_rewrite_prompt,
    render_section_prompt,
    render_section_prompt_compressed,
    render_section_repair_prompt,
)
from quality import (
    CompositeFidelityVerifier,
    EntityPresenceChecker,
    FidelityNumericFactChecker,
    FidelityVerifier,
    GenerationNumericFactChecker,
    NoOpVerifier,
    NumericFact,
    NumericFactChecker,
    OutlineCoverageChecker,
    RepetitionAndDriftChecker,
    StrictConsistencyEditGuard,
    TerminologyConsistencyChecker,
    TokenJaccardVerifier,
)
from tokenization import Tokenizer, WhitespaceTokenizer, take_last_tokens

__all__ = [
    "config",
    "protocols",
    "types",
    "Tokenizer",
    "WhitespaceTokenizer",
    "take_last_tokens",
    "LLMTask",
    "LLMRequest",
    "LLMModel",
    "RewriteModel",
    "RewriteModelFromLLM",
    "PromptLanguage",
    "RewriteRequest",
    "GenerationConfig",
    "SectionSpec",
    "GenerationPlan",
    "GenerationState",
    "QualityReport",
    "GenerationResult",
    "PipelineConfig",
    "ChunkWiseRephrasePipeline",
    "ChunkWiseGenerationPipeline",
    "_longest_overlap",
    "stitch_rewritten_chunks",
    "render_rewrite_prompt",
    "render_plan_prompt",
    "render_section_prompt",
    "render_section_prompt_compressed",
    "render_section_repair_prompt",
    "render_consistency_prompt",
    "FidelityVerifier",
    "NoOpVerifier",
    "TokenJaccardVerifier",
    "NumericFact",
    "FidelityNumericFactChecker",
    "CompositeFidelityVerifier",
    "EntityPresenceChecker",
    "GenerationNumericFactChecker",
    "NumericFactChecker",
    "OutlineCoverageChecker",
    "TerminologyConsistencyChecker",
    "RepetitionAndDriftChecker",
    "StrictConsistencyEditGuard",
    "OpenAIBackendConfig",
    "OpenAILLMModel",
    "OpenAIRewriteModel",
    "detect_dominant_script",
    "get_adaptive_length",
    "split_into_structural_units",
    "split_into_lines",
    "split_into_char_chunks",
    "split_document_into_chunks",
    "initialize_state",
    "update_state",
]

from __future__ import annotations

from . import base, evaluation, fidelity, generation
from .evaluation import (
    ABComparison,
    CoherenceMetrics,
    compare_chunked_vs_one_shot,
    evaluate_generation_coherence,
)
from .fidelity import (
    CompositeFidelityVerifier,
    FidelityVerifier,
    NoOpVerifier,
    NumericFact,
    NumericFactChecker as FidelityNumericFactChecker,
    TokenJaccardVerifier,
)
from .generation import (
    EntityPresenceChecker,
    NumericFactChecker as GenerationNumericFactChecker,
    OutlineCoverageChecker,
    RepetitionAndDriftChecker,
    StrictConsistencyEditGuard,
    TerminologyConsistencyChecker,
    TransitionContractChecker,
)

# Keep the short alias for compatibility with previous generation_quality usage.
NumericFactChecker = GenerationNumericFactChecker

__all__ = [
    "base",
    "evaluation",
    "fidelity",
    "generation",
    "CoherenceMetrics",
    "ABComparison",
    "evaluate_generation_coherence",
    "compare_chunked_vs_one_shot",
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
    "TransitionContractChecker",
    "StrictConsistencyEditGuard",
]

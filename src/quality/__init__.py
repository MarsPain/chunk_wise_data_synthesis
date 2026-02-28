from __future__ import annotations

from . import base, fidelity, generation
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
)

# Keep the short alias for compatibility with previous generation_quality usage.
NumericFactChecker = GenerationNumericFactChecker

__all__ = [
    "base",
    "fidelity",
    "generation",
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
]

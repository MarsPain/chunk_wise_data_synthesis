from __future__ import annotations

"""Legacy fidelity module kept for backward compatibility.

Canonical implementation now lives in `quality.fidelity`.
"""

from quality.fidelity import (
    CompositeFidelityVerifier,
    FidelityVerifier,
    NoOpVerifier,
    NumericFact,
    NumericFactChecker,
    TokenJaccardVerifier,
)

__all__ = [
    "FidelityVerifier",
    "NoOpVerifier",
    "TokenJaccardVerifier",
    "NumericFact",
    "NumericFactChecker",
    "CompositeFidelityVerifier",
]

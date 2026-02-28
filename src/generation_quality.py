from __future__ import annotations

"""Legacy generation quality module kept for backward compatibility.

Canonical implementation now lives in `quality.generation`.
"""

from quality.generation import (
    EntityPresenceChecker,
    NumericFactChecker,
    OutlineCoverageChecker,
    RepetitionAndDriftChecker,
    StrictConsistencyEditGuard,
    TerminologyConsistencyChecker,
    _is_key_point_covered,
    _token_jaccard,
    _tokenize,
    _words_in_order,
)

__all__ = [
    "EntityPresenceChecker",
    "NumericFactChecker",
    "OutlineCoverageChecker",
    "TerminologyConsistencyChecker",
    "RepetitionAndDriftChecker",
    "StrictConsistencyEditGuard",
    "_tokenize",
    "_token_jaccard",
    "_words_in_order",
    "_is_key_point_covered",
]

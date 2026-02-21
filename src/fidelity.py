from __future__ import annotations

import re
from typing import NamedTuple, Protocol

from tokenizer import Tokenizer, WhitespaceTokenizer


class FidelityVerifier(Protocol):
    """Protocol for fidelity verifiers used in Rephrase Pipeline."""
    
    def score(self, source_text: str, rewritten_text: str) -> float:
        """Return fidelity score between 0.0 and 1.0."""
        ...

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        """Return list of specific fidelity issues found."""
        ...


class NoOpVerifier:
    """No-op verifier that always returns perfect fidelity."""
    
    def score(self, source_text: str, rewritten_text: str) -> float:
        return 1.0

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        return []


class TokenJaccardVerifier:
    """Verifier based on token Jaccard similarity."""
    
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or WhitespaceTokenizer()

    def score(self, source_text: str, rewritten_text: str) -> float:
        source_tokens = set(self._tokenizer.encode(source_text))
        rewritten_tokens = set(self._tokenizer.encode(rewritten_text))
        if not source_tokens and not rewritten_tokens:
            return 1.0
        if not source_tokens or not rewritten_tokens:
            return 0.0
        intersection = len(source_tokens & rewritten_tokens)
        union = len(source_tokens | rewritten_tokens)
        if union == 0:
            return 0.0
        return intersection / union

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        return []


class NumericFact(NamedTuple):
    """Represents a numeric fact extracted from text."""
    value: str
    context: str
    fact_type: str  # 'year', 'percentage', 'quantity', 'version', etc.


class NumericFactChecker:
    """Detects missing or altered numeric facts between source and target text.
    
    Core numeric fact checking logic used by both:
    - Rephrase Pipeline (via FidelityVerifier interface)
    - Generation Pipeline (via find_missing() with detailed context)
    """
    
    # Regex patterns for different numeric fact types
    _YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')
    _PERCENT_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\s*%')
    _DECIMAL_PERCENT_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\s*(?:percent|percentage|pct)\b', re.IGNORECASE)
    _CURRENCY_PATTERN = re.compile(r'(?:\$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T|K))?\b', re.IGNORECASE)
    _QUANTITY_PATTERN = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand|M|B|T|K)\b', re.IGNORECASE)
    _RATIO_PATTERN = re.compile(r'\b\d+\s*:\s*\d+\b')
    _VERSION_PATTERN = re.compile(r'\bv\d+(?:\.\d+){1,2}\b|\b\d+\.\d+\.\d+\b')
    
    def __init__(
        self,
        context_window: int = 30,
        numeric_penalty: float = 0.0,
    ) -> None:
        """Initialize checker.
        
        Args:
            context_window: Number of characters before/after the match for context.
            numeric_penalty: Penalty per missing numeric fact for fidelity scoring.
                           If 0, score() returns 1.0 (checker mode, no penalty).
        """
        self._context_window = context_window
        self._numeric_penalty = numeric_penalty
    
    def find_missing(self, source_text: str, target_text: str) -> list[NumericFact]:
        """Find numeric facts present in source but missing in target.
        
        Args:
            source_text: The original/reference text.
            target_text: The text to check against.
            
        Returns:
            List of NumericFact objects that are missing from target_text.
        """
        missing: list[NumericFact] = []
        
        # Extract all numeric facts from both texts
        source_facts = self._extract_facts(source_text)
        target_facts = self._extract_facts(target_text)
        
        # Check for missing years (strict - every year must be preserved)
        source_years = {f for f in source_facts if f.fact_type == 'year'}
        target_years = {f for f in target_facts if f.fact_type == 'year'}
        for year_fact in source_years:
            if not self._has_matching_fact(year_fact, target_years, exact=True):
                missing.append(year_fact)
        
        # Check for missing percentages
        source_percents = {f for f in source_facts if f.fact_type == 'percentage'}
        target_percents = {f for f in target_facts if f.fact_type == 'percentage'}
        for pct_fact in source_percents:
            if not self._has_matching_fact(pct_fact, target_percents, exact=True):
                missing.append(pct_fact)
        
        # Check for missing quantities (with fuzzy matching)
        source_quantities = {f for f in source_facts if f.fact_type == 'quantity'}
        target_quantities = {f for f in target_facts if f.fact_type == 'quantity'}
        for qty_fact in source_quantities:
            if not self._has_matching_fact(qty_fact, target_quantities, exact=False):
                missing.append(qty_fact)
        
        # Check for missing versions
        source_versions = {f for f in source_facts if f.fact_type == 'version'}
        target_versions = {f for f in target_facts if f.fact_type == 'version'}
        for ver_fact in source_versions:
            if not self._has_matching_fact(ver_fact, target_versions, exact=True):
                missing.append(ver_fact)
        
        return missing
    
    def _extract_facts(self, text: str) -> set[NumericFact]:
        """Extract all numeric facts from text."""
        facts: set[NumericFact] = set()
        
        # Extract years
        for match in self._YEAR_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group(), context, 'year'))
        
        # Extract percentages (symbol form)
        for match in self._PERCENT_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group().lower().replace(' ', ''), context, 'percentage'))
        
        # Extract percentages (word form)
        for match in self._DECIMAL_PERCENT_PATTERN.finditer(text):
            value = match.group().lower().replace('percent', '%').replace('percentage', '%').replace('pct', '%').replace(' ', '')
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(value, context, 'percentage'))
        
        # Extract quantities
        for match in self._QUANTITY_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group().lower(), context, 'quantity'))
        
        # Extract versions
        for match in self._VERSION_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group(), context, 'version'))
        
        return facts
    
    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around a match."""
        context_start = max(0, start - self._context_window)
        context_end = min(len(text), end + self._context_window)
        return text[context_start:context_end].strip()
    
    def _has_matching_fact(
        self, 
        fact: NumericFact, 
        candidates: set[NumericFact],
        exact: bool = True
    ) -> bool:
        """Check if a matching fact exists in candidates."""
        for candidate in candidates:
            if exact:
                if self._normalize_value(fact.value) == self._normalize_value(candidate.value):
                    return True
            else:
                if self._values_similar(fact.value, candidate.value):
                    return True
        return False
    
    def _normalize_value(self, value: str) -> str:
        """Normalize a numeric value for comparison."""
        value = value.lower().replace(',', '').replace(' ', '')
        if value.startswith('v') and value[1:].replace('.', '').isdigit():
            value = value[1:]
        return value
    
    def _values_similar(self, val1: str, val2: str) -> bool:
        """Check if two numeric values are similar (for quantities)."""
        num1 = self._extract_number(val1)
        num2 = self._extract_number(val2)
        
        if num1 is not None and num2 is not None:
            if num2 > 0:
                return abs(num1 - num2) / num2 < 0.05
            return num1 == num2
        
        return self._normalize_value(val1) == self._normalize_value(val2)
    
    def _extract_number(self, value: str) -> float | None:
        """Extract numeric value from string like '1.5 million' or '2,000K'."""
        try:
            value = value.lower().replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '')
            
            multipliers = {'million': 1e6, 'billion': 1e9, 'trillion': 1e12,
                          'm': 1e6, 'b': 1e9, 't': 1e12, 'k': 1e3, 'thousand': 1e3}
            
            for suffix, multiplier in multipliers.items():
                if suffix in value:
                    num_part = value.replace(suffix, '').strip()
                    return float(num_part) * multiplier
            
            return float(value)
        except (ValueError, TypeError):
            return None

    # FidelityVerifier protocol methods - for use in Rephrase Pipeline
    
    def score(self, source_text: str, rewritten_text: str) -> float:
        """Calculate fidelity score with optional numeric fact penalties."""
        if self._numeric_penalty <= 0:
            return 1.0
        
        missing_count = len(self.find_missing(source_text, rewritten_text))
        if missing_count == 0:
            return 1.0
        
        return max(0.0, 1.0 - (missing_count * self._numeric_penalty))
    
    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        """Return list of missing numeric fact issues."""
        missing_facts = self.find_missing(source_text, rewritten_text)
        return [f"{f.fact_type.capitalize()} {f.value} missing" for f in missing_facts]


class CompositeFidelityVerifier:
    """Combines multiple verifiers with configurable weights."""

    def __init__(
        self,
        verifiers: list[tuple[FidelityVerifier, float]],
    ) -> None:
        """Initialize with list of (verifier, weight) tuples."""
        self._verifiers = verifiers
        self._issues: list[str] = []

    def score(self, source_text: str, rewritten_text: str) -> float:
        """Calculate weighted composite score."""
        total_score = 0.0
        total_weight = 0.0
        self._issues = []

        for verifier, weight in self._verifiers:
            score = verifier.score(source_text, rewritten_text)
            total_score += score * weight
            total_weight += weight
            
            issues = verifier.get_issues(source_text, rewritten_text)
            self._issues.extend(issues)

        if total_weight == 0:
            return 1.0
        return total_score / total_weight

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        """Return aggregated issues from all verifiers."""
        _ = self.score(source_text, rewritten_text)
        return self._issues

from __future__ import annotations

import re
from typing import NamedTuple, Protocol

from tokenization import Tokenizer, WhitespaceTokenizer


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
    fact_type: str


class NumericFactChecker:
    """Detects missing or altered numeric facts between source and target text."""

    _YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
    _PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*%")
    _DECIMAL_PERCENT_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:percent|percentage|pct)\b",
        re.IGNORECASE,
    )
    _CURRENCY_PATTERN = re.compile(
        r"(?:\$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T|K))?\b",
        re.IGNORECASE,
    )
    _QUANTITY_PATTERN = re.compile(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand|M|B|T|K)\b",
        re.IGNORECASE,
    )
    _RATIO_PATTERN = re.compile(r"\b\d+\s*:\s*\d+\b")
    _VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+){1,2}\b|\b\d+\.\d+\.\d+\b")

    def __init__(
        self,
        context_window: int = 30,
        numeric_penalty: float = 0.0,
    ) -> None:
        self._context_window = context_window
        self._numeric_penalty = numeric_penalty

    def find_missing(self, source_text: str, target_text: str) -> list[NumericFact]:
        missing: list[NumericFact] = []
        source_facts = self._extract_facts(source_text)
        target_facts = self._extract_facts(target_text)

        source_years = {fact for fact in source_facts if fact.fact_type == "year"}
        target_years = {fact for fact in target_facts if fact.fact_type == "year"}
        for year_fact in source_years:
            if not self._has_matching_fact(year_fact, target_years, exact=True):
                missing.append(year_fact)

        source_percents = {fact for fact in source_facts if fact.fact_type == "percentage"}
        target_percents = {fact for fact in target_facts if fact.fact_type == "percentage"}
        for percent_fact in source_percents:
            if not self._has_matching_fact(percent_fact, target_percents, exact=True):
                missing.append(percent_fact)

        source_quantities = {fact for fact in source_facts if fact.fact_type == "quantity"}
        target_quantities = {fact for fact in target_facts if fact.fact_type == "quantity"}
        for quantity_fact in source_quantities:
            if not self._has_matching_fact(quantity_fact, target_quantities, exact=False):
                missing.append(quantity_fact)

        source_versions = {fact for fact in source_facts if fact.fact_type == "version"}
        target_versions = {fact for fact in target_facts if fact.fact_type == "version"}
        for version_fact in source_versions:
            if not self._has_matching_fact(version_fact, target_versions, exact=True):
                missing.append(version_fact)

        return missing

    def _extract_facts(self, text: str) -> set[NumericFact]:
        facts: set[NumericFact] = set()

        for match in self._YEAR_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group(), context, "year"))

        for match in self._PERCENT_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group().lower().replace(" ", ""), context, "percentage"))

        for match in self._DECIMAL_PERCENT_PATTERN.finditer(text):
            value = (
                match.group()
                .lower()
                .replace("percent", "%")
                .replace("percentage", "%")
                .replace("pct", "%")
                .replace(" ", "")
            )
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(value, context, "percentage"))

        for match in self._QUANTITY_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group().lower(), context, "quantity"))

        for match in self._VERSION_PATTERN.finditer(text):
            context = self._extract_context(text, match.start(), match.end())
            facts.add(NumericFact(match.group(), context, "version"))

        return facts

    def _extract_context(self, text: str, start: int, end: int) -> str:
        context_start = max(0, start - self._context_window)
        context_end = min(len(text), end + self._context_window)
        return text[context_start:context_end].strip()

    def _has_matching_fact(
        self,
        fact: NumericFact,
        candidates: set[NumericFact],
        exact: bool = True,
    ) -> bool:
        for candidate in candidates:
            if exact:
                if self._normalize_value(fact.value) == self._normalize_value(candidate.value):
                    return True
            else:
                if self._values_similar(fact.value, candidate.value):
                    return True
        return False

    def _normalize_value(self, value: str) -> str:
        normalized = value.lower().replace(",", "").replace(" ", "")
        if normalized.startswith("v") and normalized[1:].replace(".", "").isdigit():
            normalized = normalized[1:]
        return normalized

    def _values_similar(self, left: str, right: str) -> bool:
        left_number = self._extract_number(left)
        right_number = self._extract_number(right)

        if left_number is not None and right_number is not None:
            if right_number > 0:
                return abs(left_number - right_number) / right_number < 0.05
            return left_number == right_number

        return self._normalize_value(left) == self._normalize_value(right)

    def _extract_number(self, value: str) -> float | None:
        try:
            normalized = (
                value.lower()
                .replace(",", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
                .replace("¥", "")
            )
            multipliers = {
                "million": 1e6,
                "billion": 1e9,
                "trillion": 1e12,
                "m": 1e6,
                "b": 1e9,
                "t": 1e12,
                "k": 1e3,
                "thousand": 1e3,
            }

            for suffix, multiplier in multipliers.items():
                if suffix in normalized:
                    number_part = normalized.replace(suffix, "").strip()
                    return float(number_part) * multiplier

            return float(normalized)
        except (ValueError, TypeError):
            return None

    def score(self, source_text: str, rewritten_text: str) -> float:
        if self._numeric_penalty <= 0:
            return 1.0

        missing_count = len(self.find_missing(source_text, rewritten_text))
        if missing_count == 0:
            return 1.0

        return max(0.0, 1.0 - (missing_count * self._numeric_penalty))

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        missing_facts = self.find_missing(source_text, rewritten_text)
        return [f"{fact.fact_type.capitalize()} {fact.value} missing" for fact in missing_facts]


class CompositeFidelityVerifier:
    """Combines multiple verifiers with configurable weights."""

    def __init__(
        self,
        verifiers: list[tuple[FidelityVerifier, float]],
    ) -> None:
        self._verifiers = verifiers
        self._issues: list[str] = []

    def score(self, source_text: str, rewritten_text: str) -> float:
        total_score = 0.0
        total_weight = 0.0
        self._issues = []

        for verifier, weight in self._verifiers:
            score = verifier.score(source_text, rewritten_text)
            total_score += score * weight
            total_weight += weight
            self._issues.extend(verifier.get_issues(source_text, rewritten_text))

        if total_weight == 0:
            return 1.0
        return total_score / total_weight

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        _ = self.score(source_text, rewritten_text)
        return self._issues


__all__ = [
    "FidelityVerifier",
    "NoOpVerifier",
    "TokenJaccardVerifier",
    "NumericFact",
    "NumericFactChecker",
    "CompositeFidelityVerifier",
]

from __future__ import annotations

import re
from dataclasses import dataclass

from generation_types import GenerationPlan, SectionSpec
from quality.base import _token_jaccard, _tokenize, _words_in_order
from quality.fidelity import NumericFactChecker as _FidelityNumericFactChecker


def _is_key_point_covered(key_point: str, text: str) -> bool:
    key_point_lower = key_point.strip().lower()
    text_lower = text.lower()
    if not key_point_lower:
        return True
    if key_point_lower in text_lower:
        return True
    point_tokens = _tokenize(key_point)
    text_tokens = _tokenize(text)
    if not point_tokens:
        return True
    overlap = len(point_tokens & text_tokens)
    required = 1 if len(point_tokens) == 1 else min(2, len(point_tokens))
    return overlap >= required


class EntityPresenceChecker:
    """Ensures required entities appear in section outputs."""

    def find_missing(
        self,
        plan: GenerationPlan,
        section_outputs: list[str],
    ) -> list[str]:
        missing: list[str] = []

        for idx, (section, text) in enumerate(zip(plan.sections, section_outputs)):
            for entity in self.missing_entities(section=section, text=text):
                missing.append(
                    f"Section {idx + 1} ('{section.title}') missing required entity: '{entity}'"
                )

        return missing

    def missing_entities(self, section: SectionSpec, text: str) -> list[str]:
        text_lower = text.lower()
        missing: list[str] = []
        for entity in section.required_entities:
            if not self._entity_present(entity.lower(), text_lower):
                missing.append(entity)
        return missing

    def _entity_present(self, entity: str, text: str) -> bool:
        if entity in text:
            return True
        if entity.replace(" ", "-") in text:
            return True
        if entity.replace(" ", "_") in text:
            return True
        words = entity.split()
        if len(words) > 1:
            return _words_in_order(words, text)
        return False


class NumericFactChecker:
    """Generation-specific wrapper around fidelity numeric fact checks."""

    def __init__(self, context_window: int = 30) -> None:
        self._core_checker = _FidelityNumericFactChecker(context_window=context_window)

    def find_missing(self, source_text: str, generated_text: str) -> list[str]:
        missing_facts = self._core_checker.find_missing(source_text, generated_text)
        return [
            (
                f"{fact.fact_type.capitalize()} '{fact.value}' may be missing "
                f"(was in: '...{fact.context}...')"
            )
            for fact in missing_facts
        ]


class OutlineCoverageChecker:
    def find_missing(self, plan: GenerationPlan, text: str) -> list[str]:
        missing: list[str] = []
        for section in plan.sections:
            for key_point in section.key_points:
                if not _is_key_point_covered(key_point=key_point, text=text):
                    missing.append(key_point)
        return missing


class TerminologyConsistencyChecker:
    def find_issues(self, plan: GenerationPlan, text: str) -> list[str]:
        issues: list[str] = []
        lowered_text = text.lower()
        for source, preferred in plan.terminology_preferences.items():
            source_lower = source.lower()
            preferred_lower = preferred.lower()
            if source_lower == preferred_lower:
                continue
            if source_lower in lowered_text and preferred_lower not in lowered_text:
                issues.append(f"Prefer '{preferred}' over '{source}'.")
        return issues


class RepetitionAndDriftChecker:
    def __init__(
        self,
        repetition_threshold: float = 0.92,
        drift_overlap_threshold: float = 0.05,
        min_tokens_for_drift: int = 8,
    ) -> None:
        self._repetition_threshold = repetition_threshold
        self._drift_overlap_threshold = drift_overlap_threshold
        self._min_tokens_for_drift = min_tokens_for_drift

    def find_issues(
        self,
        plan: GenerationPlan,
        section_outputs: list[str],
    ) -> tuple[list[str], list[str]]:
        repetition_issues: list[str] = []
        drift_issues: list[str] = []

        for idx in range(1, len(section_outputs)):
            score = _token_jaccard(section_outputs[idx - 1], section_outputs[idx])
            if score >= self._repetition_threshold:
                repetition_issues.append(
                    f"Section {idx} and section {idx + 1} are highly repetitive (score={score:.2f})."
                )

        allowed_tokens = _tokenize(plan.topic)
        allowed_tokens.update(_tokenize(plan.objective))
        allowed_tokens.update(_tokenize(plan.audience))
        for section in plan.sections:
            allowed_tokens.update(_tokenize(section.title))
            for point in section.key_points:
                allowed_tokens.update(_tokenize(point))
            for entity in section.required_entities:
                allowed_tokens.update(_tokenize(entity))
        for source, preferred in plan.terminology_preferences.items():
            allowed_tokens.update(_tokenize(source))
            allowed_tokens.update(_tokenize(preferred))

        for idx, section_text in enumerate(section_outputs):
            section_tokens = _tokenize(section_text)
            if len(section_tokens) < self._min_tokens_for_drift:
                continue
            overlap = len(section_tokens & allowed_tokens) / len(section_tokens)
            if overlap < self._drift_overlap_threshold:
                drift_issues.append(
                    f"Section {idx + 1} may drift from plan topics (overlap={overlap:.2f})."
                )

        return repetition_issues, drift_issues


@dataclass(frozen=True)
class StrictConsistencyEditGuard:
    min_token_jaccard: float = 0.75
    max_length_ratio: float = 1.3
    min_length_ratio: float = 0.7
    max_added_sentences: int = 2

    def apply(self, original_text: str, candidate_text: str) -> tuple[str, bool]:
        original = original_text.strip()
        candidate = candidate_text.strip()
        if not original:
            return candidate, False
        if not candidate:
            return original, True

        similarity = _token_jaccard(original, candidate)
        if similarity < self.min_token_jaccard:
            return original, True

        original_tokens = list(_tokenize(original))
        candidate_tokens = list(_tokenize(candidate))
        token_count = max(len(original_tokens), 1)
        length_ratio = len(candidate_tokens) / token_count
        if length_ratio < self.min_length_ratio or length_ratio > self.max_length_ratio:
            return original, True

        original_sentences = [part.strip() for part in re.split(r"[.!?]+", original) if part.strip()]
        candidate_sentences = [part.strip() for part in re.split(r"[.!?]+", candidate) if part.strip()]
        if len(candidate_sentences) - len(original_sentences) > self.max_added_sentences:
            return original, True

        return candidate, False


__all__ = [
    "_tokenize",
    "_token_jaccard",
    "_words_in_order",
    "_is_key_point_covered",
    "EntityPresenceChecker",
    "NumericFactChecker",
    "OutlineCoverageChecker",
    "TerminologyConsistencyChecker",
    "RepetitionAndDriftChecker",
    "StrictConsistencyEditGuard",
]

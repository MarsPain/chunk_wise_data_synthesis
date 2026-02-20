from __future__ import annotations

import re
from typing import Iterable

from generation_types import GenerationPlan, GenerationState, SectionSpec

_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def _is_key_point_covered(key_point: str, text: str) -> bool:
    normalized_point = key_point.strip().lower()
    normalized_text = text.lower()
    if not normalized_point:
        return True
    if normalized_point in normalized_text:
        return True

    point_tokens = _tokenize(key_point)
    text_tokens = _tokenize(text)
    if not point_tokens:
        return True

    overlap = len(point_tokens & text_tokens)
    required_overlap = 1 if len(point_tokens) == 1 else min(len(point_tokens), 2)
    return overlap >= required_overlap


def _merge_unique(existing: Iterable[str], new_items: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(existing) + list(new_items):
        value = item.strip()
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        merged.append(value)
    return merged


def _extract_entities(section_text: str, section_spec: SectionSpec) -> list[str]:
    inferred = [match.group(0) for match in _ENTITY_PATTERN.finditer(section_text)]
    for entity in section_spec.required_entities:
        if entity.lower() in section_text.lower():
            inferred.append(entity)
    return _merge_unique([], inferred)


def initialize_state(plan: GenerationPlan) -> GenerationState:
    return GenerationState.from_plan(plan)


def update_state(
    state: GenerationState,
    plan: GenerationPlan,
    section_spec: SectionSpec,
    section_text: str,
) -> GenerationState:
    entities = _merge_unique(
        state.known_entities,
        _extract_entities(section_text=section_text, section_spec=section_spec),
    )

    terminology_map = dict(state.terminology_map)
    for term, preferred in plan.terminology_preferences.items():
        if preferred.lower() in section_text.lower() or term.lower() in section_text.lower():
            terminology_map[term] = preferred

    timeline = list(state.timeline)
    for match in _YEAR_PATTERN.finditer(section_text):
        year = match.group(0)
        if year not in timeline:
            timeline.append(year)

    covered = list(state.covered_key_points)
    remaining = list(state.remaining_key_points)
    for key_point in section_spec.key_points:
        if _is_key_point_covered(key_point=key_point, text=section_text):
            if key_point not in covered:
                covered.append(key_point)
            if key_point in remaining:
                remaining.remove(key_point)

    return GenerationState(
        known_entities=entities,
        terminology_map=terminology_map,
        timeline=timeline,
        covered_key_points=covered,
        remaining_key_points=remaining,
    )


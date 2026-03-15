from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from generation_types import GenerationPlan
from quality.generation import OutlineCoverageChecker, TerminologyConsistencyChecker

_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_TRANSITION_MARKERS = (
    "however",
    "therefore",
    "meanwhile",
    "next",
    "in the next section",
    "building on",
    "based on",
    "同时",
    "此外",
    "因此",
    "接下来",
    "基于",
    "在下一节",
)


@dataclass(frozen=True)
class CoherenceMetrics:
    boundary_coherence: float
    repetition_risk: float
    drift_risk: float
    terminology_issue_count: int
    coverage_missing_count: int
    aggregate_score: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "boundary_coherence": self.boundary_coherence,
            "repetition_risk": self.repetition_risk,
            "drift_risk": self.drift_risk,
            "terminology_issue_count": self.terminology_issue_count,
            "coverage_missing_count": self.coverage_missing_count,
            "aggregate_score": self.aggregate_score,
        }


@dataclass(frozen=True)
class ABComparison:
    case_id: str
    winner: Literal["chunked", "one_shot", "tie"]
    score_delta: float
    chunked: CoherenceMetrics
    one_shot: CoherenceMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "winner": self.winner,
            "score_delta": self.score_delta,
            "chunked": self.chunked.to_dict(),
            "one_shot": self.one_shot.to_dict(),
        }


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _extract_units(text: str) -> list[str]:
    words = [token.lower() for token in _WORD_PATTERN.findall(text)]
    cjk_chars = _CJK_PATTERN.findall(text)
    cjk_bigrams = [
        f"{cjk_chars[idx]}{cjk_chars[idx + 1]}"
        for idx in range(len(cjk_chars) - 1)
    ]
    return words + cjk_bigrams


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _has_transition_marker(text: str) -> bool:
    lowered = text.strip().lower()
    return any(marker in lowered[:80] for marker in _TRANSITION_MARKERS)


def _boundary_coherence(section_outputs: list[str], window_size: int = 36) -> float:
    if len(section_outputs) <= 1:
        return 1.0

    scores: list[float] = []
    for idx in range(1, len(section_outputs)):
        prev_units = _extract_units(section_outputs[idx - 1])
        next_units = _extract_units(section_outputs[idx])
        tail = prev_units[-window_size:]
        head = next_units[:window_size]
        local_overlap = _jaccard(tail, head)
        adjacent_repetition = _jaccard(prev_units, next_units)
        # Pure lexical overlap across boundaries can come from repetition.
        score = local_overlap * (1.0 - 0.6 * adjacent_repetition)
        if _has_transition_marker(section_outputs[idx]):
            score = min(1.0, score + 0.15)
        scores.append(score)

    return sum(scores) / max(len(scores), 1)


def _repetition_risk(section_outputs: list[str]) -> float:
    if len(section_outputs) <= 1:
        return 0.0

    scores: list[float] = []
    for idx in range(1, len(section_outputs)):
        previous = _extract_units(section_outputs[idx - 1])
        current = _extract_units(section_outputs[idx])
        scores.append(_jaccard(previous, current))

    return sum(scores) / max(len(scores), 1)


def _allowed_units(plan: GenerationPlan) -> set[str]:
    source_texts: list[str] = [plan.topic, plan.objective, plan.audience, plan.tone]
    for section in plan.sections:
        source_texts.append(section.title)
        source_texts.extend(section.key_points)
        source_texts.extend(section.required_entities)
        source_texts.extend(section.constraints)
    for source, preferred in plan.terminology_preferences.items():
        source_texts.append(source)
        source_texts.append(preferred)

    merged = " ".join(source_texts)
    return set(_extract_units(merged))


def _drift_risk(plan: GenerationPlan, section_outputs: list[str]) -> float:
    if not section_outputs:
        return 1.0

    allowed = _allowed_units(plan)
    if not allowed:
        return 0.0

    risks: list[float] = []
    for section_text in section_outputs:
        units = set(_extract_units(section_text))
        if not units:
            risks.append(1.0)
            continue
        overlap = len(units & allowed) / len(units)
        risks.append(1.0 - overlap)

    return sum(risks) / max(len(risks), 1)


def evaluate_generation_coherence(
    plan: GenerationPlan,
    section_outputs: list[str],
    final_text: str | None = None,
) -> CoherenceMetrics:
    normalized_sections = [part.strip() for part in section_outputs if part.strip()]
    assembled_text = final_text.strip() if final_text else "\n\n".join(normalized_sections).strip()

    boundary = _clamp01(_boundary_coherence(normalized_sections))
    repetition = _clamp01(_repetition_risk(normalized_sections))
    drift = _clamp01(_drift_risk(plan, normalized_sections))

    terminology_checker = TerminologyConsistencyChecker()
    outline_checker = OutlineCoverageChecker()
    terminology_issues = terminology_checker.find_issues(plan=plan, text=assembled_text)
    coverage_missing = outline_checker.find_missing(plan=plan, text=assembled_text)

    terminology_score = 1.0 / (1.0 + len(terminology_issues))
    coverage_score = 1.0 / (1.0 + len(coverage_missing))
    aggregate = _clamp01(
        0.35 * boundary
        + 0.20 * (1.0 - repetition)
        + 0.20 * (1.0 - drift)
        + 0.15 * terminology_score
        + 0.10 * coverage_score
    )

    return CoherenceMetrics(
        boundary_coherence=boundary,
        repetition_risk=repetition,
        drift_risk=drift,
        terminology_issue_count=len(terminology_issues),
        coverage_missing_count=len(coverage_missing),
        aggregate_score=aggregate,
    )


def compare_chunked_vs_one_shot(
    case_id: str,
    chunked_metrics: CoherenceMetrics,
    one_shot_metrics: CoherenceMetrics,
    tie_epsilon: float = 1e-6,
) -> ABComparison:
    delta = chunked_metrics.aggregate_score - one_shot_metrics.aggregate_score
    if abs(delta) <= tie_epsilon:
        winner: Literal["chunked", "one_shot", "tie"] = "tie"
    elif delta > 0:
        winner = "chunked"
    else:
        winner = "one_shot"

    return ABComparison(
        case_id=case_id,
        winner=winner,
        score_delta=delta,
        chunked=chunked_metrics,
        one_shot=one_shot_metrics,
    )


__all__ = [
    "CoherenceMetrics",
    "ABComparison",
    "evaluate_generation_coherence",
    "compare_chunked_vs_one_shot",
]

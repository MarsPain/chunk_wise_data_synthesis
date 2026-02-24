from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("expected list value")
    return [str(item).strip() for item in raw if str(item).strip()]


def _extract_json_object(raw: str) -> str:
    """Extract JSON object from model output, handling markdown code blocks and reasoning content."""
    import re
    
    raw = raw.strip()
    
    # Handle markdown code blocks: ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        # Remove opening fence
        first_newline = raw.find("\n")
        if first_newline > 0:
            raw = raw[first_newline:].lstrip()
        # Remove closing fence
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
    
    # Remove common thinking/reasoning tags (for models that output thinking in content)
    # Pattern 1: <think>...</think>
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Pattern 2: <thinking>...</thinking>
    raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
    # Pattern 3: <reasoning>...</reasoning>
    raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
    
    raw = raw.strip()
    
    # Strategy 1: Try to find a balanced JSON object by tracking brace depth
    # This handles nested objects correctly
    start = -1
    depth = 0
    end = -1
    in_double_string = False
    in_single_string = False
    escape_next = False
    
    for i, char in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and (in_double_string or in_single_string):
            escape_next = True
            continue
        if char == '"' and not in_single_string:
            in_double_string = not in_double_string
            continue
        if char == "'" and not in_double_string:
            in_single_string = not in_single_string
            continue
        if in_double_string or in_single_string:
            continue
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    end = i
                    break
    
    if start >= 0 and end >= start:
        candidate = raw[start : end + 1]
        # Validate it's actually valid JSON
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # Fall through to next strategy
    
    # Strategy 2: Look for JSON between first '{' and last '}'
    # This handles cases where there's trailing text after the JSON
    first_brace = raw.find('{')
    last_brace = raw.rfind('}')
    
    if first_brace >= 0 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # Fall through to error
    
    # Check if output appears truncated (starts with { but doesn't end with })
    raw_stripped = raw.strip()
    is_truncated = (
        raw_stripped.startswith('{') 
        and not raw_stripped.endswith('}')
        and raw_stripped.count('{') > raw_stripped.count('}')
    )
    
    if is_truncated:
        raise ValueError(
            f"model output appears to be truncated (incomplete JSON). "
            f"This usually means the output exceeded max_tokens limit. "
            f"Try increasing max_new_tokens. Raw output: {raw[:500]}"
        )
    
    raise ValueError(f"model output does not contain a valid JSON object. Raw output: {raw[:500]}")


@dataclass(frozen=True)
class SectionSpec:
    title: str
    key_points: list[str]
    required_entities: list[str]
    constraints: list[str]
    target_length: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SectionSpec":
        title = str(payload.get("title", "")).strip()
        target_length = int(payload.get("target_length", 0) or 0)
        return cls(
            title=title,
            key_points=_as_string_list(payload.get("key_points")),
            required_entities=_as_string_list(payload.get("required_entities")),
            constraints=_as_string_list(payload.get("constraints")),
            target_length=target_length,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "key_points": list(self.key_points),
            "required_entities": list(self.required_entities),
            "constraints": list(self.constraints),
            "target_length": self.target_length,
        }


@dataclass(frozen=True)
class GenerationPlan:
    topic: str
    objective: str
    audience: str
    tone: str
    target_total_length: int
    sections: list[SectionSpec]
    terminology_preferences: dict[str, str]
    narrative_voice: str
    do_not_include: list[str]

    @classmethod
    def from_json(cls, raw: str) -> "GenerationPlan":
        try:
            extracted = _extract_json_object(raw)
        except ValueError as exc:
            raise ValueError(f"Failed to extract JSON from model output: {exc}") from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in model output: {exc}. Extracted: {extracted[:500]}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"plan payload must be an object, got {type(payload).__name__}")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationPlan":
        raw_preferences = payload.get("terminology_preferences") or {}
        if not isinstance(raw_preferences, dict):
            raise ValueError("terminology_preferences must be an object")
        sections_payload = payload.get("sections") or []
        if not isinstance(sections_payload, list):
            raise ValueError("sections must be a list")

        plan = cls(
            topic=str(payload.get("topic", "")).strip(),
            objective=str(payload.get("objective", "")).strip(),
            audience=str(payload.get("audience", "")).strip(),
            tone=str(payload.get("tone", "")).strip(),
            target_total_length=int(payload.get("target_total_length", 0) or 0),
            sections=[
                SectionSpec.from_dict(section)
                for section in sections_payload
                if isinstance(section, dict)
            ],
            terminology_preferences={
                str(key).strip(): str(value).strip()
                for key, value in raw_preferences.items()
                if str(key).strip() and str(value).strip()
            },
            narrative_voice=str(payload.get("narrative_voice", "third-person")).strip()
            or "third-person",
            do_not_include=_as_string_list(payload.get("do_not_include")),
        )
        plan.validate()
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "objective": self.objective,
            "audience": self.audience,
            "tone": self.tone,
            "target_total_length": self.target_total_length,
            "narrative_voice": self.narrative_voice,
            "do_not_include": list(self.do_not_include),
            "terminology_preferences": dict(self.terminology_preferences),
            "sections": [section.to_dict() for section in self.sections],
        }

    def validate(self) -> None:
        if not self.topic:
            raise ValueError("plan topic must not be empty")
        if not self.objective:
            raise ValueError("plan objective must not be empty")
        if self.target_total_length <= 0:
            raise ValueError("target_total_length must be positive")
        if not self.sections:
            raise ValueError("plan must include at least one section")

        for idx, section in enumerate(self.sections):
            if not section.title:
                raise ValueError(f"section[{idx}] title must not be empty")
            if section.target_length <= 0:
                raise ValueError(f"section[{idx}] target_length must be positive")
            if not section.key_points:
                raise ValueError(f"section[{idx}] must define key_points")


@dataclass(frozen=True)
class GenerationState:
    known_entities: list[str] = field(default_factory=list)
    terminology_map: dict[str, str] = field(default_factory=dict)
    timeline: list[str] = field(default_factory=list)
    covered_key_points: list[str] = field(default_factory=list)
    remaining_key_points: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: GenerationPlan) -> "GenerationState":
        remaining: list[str] = []
        for section in plan.sections:
            for point in section.key_points:
                if point not in remaining:
                    remaining.append(point)
        return cls(
            known_entities=[],
            terminology_map=dict(plan.terminology_preferences),
            timeline=[],
            covered_key_points=[],
            remaining_key_points=remaining,
        )


@dataclass
class QualityReport:
    coverage_missing: list[str] = field(default_factory=list)
    terminology_issues: list[str] = field(default_factory=list)
    repetition_issues: list[str] = field(default_factory=list)
    drift_issues: list[str] = field(default_factory=list)
    section_warnings: list[str] = field(default_factory=list)
    entity_missing: list[str] = field(default_factory=list)
    numeric_fact_issues: list[str] = field(default_factory=list)
    consistency_pass_applied: bool = False
    consistency_pass_used_fallback: bool = False

    def has_issues(self) -> bool:
        return any(
            [
                bool(self.coverage_missing),
                bool(self.terminology_issues),
                bool(self.repetition_issues),
                bool(self.drift_issues),
            ]
        )

    def has_critical_issues(self) -> bool:
        """Critical issues include missing required entities and numeric fact errors."""
        return any(
            [
                bool(self.entity_missing),
                bool(self.numeric_fact_issues),
            ]
        )


@dataclass(frozen=True)
class GenerationResult:
    final_text: str
    plan: GenerationPlan
    section_outputs: list[str]
    final_state: GenerationState
    qc_report: QualityReport


@dataclass(frozen=True)
class GenerationConfig:
    prefix_window_tokens: int = 1200
    min_section_length_ratio: float = 0.8
    max_section_length_ratio: float = 1.2
    repetition_similarity_threshold: float = 0.92
    drift_overlap_threshold: float = 0.05
    consistency_pass_enabled: bool = True
    consistency_guard_min_token_jaccard: float = 0.75
    consistency_guard_min_length_ratio: float = 0.7
    consistency_guard_max_length_ratio: float = 1.3
    consistency_guard_max_added_sentences: int = 2

    # Section retry configuration (P0)
    max_section_retries: int = 2
    section_quality_threshold: float = 0.8
    retry_on_missing_entities: bool = True
    retry_on_length_violation: bool = False
    entity_missing_penalty: float = 0.2
    length_violation_penalty: float = 0.1
    repetition_penalty: float = 0.15

    # Prompt compression configuration
    prompt_compression_enabled: bool = True
    max_covered_points_summary_items: int = 3
    max_entities_in_prompt: int = 20
    max_timeline_entries: int = 5
    upcoming_sections_preview: int = 2
    prompt_language: Literal["en", "zh"] = "en"

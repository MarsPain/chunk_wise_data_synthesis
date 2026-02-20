from __future__ import annotations

import json

from generation_types import GenerationPlan, GenerationState, QualityReport, SectionSpec


def render_plan_prompt(
    topic: str,
    objective: str,
    target_tokens: int,
    audience: str,
    tone: str,
) -> str:
    schema_hint = {
        "topic": topic,
        "objective": objective,
        "audience": audience or "general technical audience",
        "tone": tone or "neutral technical",
        "target_total_length": target_tokens,
        "narrative_voice": "third-person",
        "do_not_include": ["unsupported claims"],
        "terminology_preferences": {"example_term": "preferred phrasing"},
        "sections": [
            {
                "title": "Section title",
                "key_points": ["point A", "point B"],
                "required_entities": ["entity A"],
                "constraints": ["constraint A"],
                "target_length": 300,
            }
        ],
    }
    return "\n\n".join(
        [
            "You are planning a long-form, section-wise generation task.",
            "CRITICAL: Output ONLY the JSON object below. Do not output any thinking, planning, or explanation text.",
            "Your response must START with '{' and END with '}'. No text before or after the JSON.",
            "Do not wrap in markdown code blocks. Do not include comments. Just the raw JSON.",
            "",
            "Build a complete generation plan with coherent sections and explicit coverage points.",
            f"Topic: {topic}",
            f"Objective: {objective}",
            f"Audience: {audience or 'general technical audience'}",
            f"Tone: {tone or 'neutral technical'}",
            f"Target total length (tokens): {target_tokens}",
            "",
            "Output schema (return ONLY a JSON object matching this structure):",
            json.dumps(schema_hint, ensure_ascii=False),
        ]
    )


def render_section_prompt(
    plan: GenerationPlan,
    state: GenerationState,
    recent_text: str,
    section_spec: SectionSpec,
) -> str:
    plan_blob = json.dumps(plan.to_dict(), ensure_ascii=False)
    state_blob = json.dumps(
        {
            "known_entities": state.known_entities,
            "terminology_map": state.terminology_map,
            "timeline": state.timeline,
            "covered_key_points": state.covered_key_points,
            "remaining_key_points": state.remaining_key_points,
        },
        ensure_ascii=False,
    )
    section_blob = json.dumps(section_spec.to_dict(), ensure_ascii=False)
    return "\n\n".join(
        [
            "You are generating one section of a long article.",
            "CRITICAL: Output ONLY the section body text. No thinking, no planning, no preamble.",
            "Rules:",
            "1) Follow plan and current section spec strictly.",
            "2) Keep terminology, entities, and timeline consistent with state.",
            "3) Avoid repeating points already covered in recent text.",
            "4) Output only the current section body text - no JSON, no markdown, no explanations.",
            "Global plan:",
            plan_blob,
            "Current state:",
            state_blob,
            f"Recent generated text:\n{recent_text or '(none)'}",
            "Current section spec:",
            section_blob,
            "Target length is approximate (allow +/-20%).",
        ]
    )


def render_consistency_prompt(
    plan: GenerationPlan,
    state: GenerationState,
    draft_text: str,
    quality_report: QualityReport,
) -> str:
    issues_blob = json.dumps(
        {
            "coverage_missing": quality_report.coverage_missing,
            "terminology_issues": quality_report.terminology_issues,
            "repetition_issues": quality_report.repetition_issues,
            "drift_issues": quality_report.drift_issues,
        },
        ensure_ascii=False,
    )
    return "\n\n".join(
        [
            "You are running a light consistency pass on a generated long-form draft.",
            "CRITICAL: Output ONLY the revised full text. No thinking, no planning, no preamble.",
            "Allowed edits only:",
            "1) fix terminology consistency",
            "2) improve transitions between sections",
            "3) add 1-2 short sentences for missing key points",
            "Do not perform major rewrites or change the structure.",
            "Plan:",
            json.dumps(plan.to_dict(), ensure_ascii=False),
            "State:",
            json.dumps(
                {
                    "known_entities": state.known_entities,
                    "terminology_map": state.terminology_map,
                    "timeline": state.timeline,
                    "remaining_key_points": state.remaining_key_points,
                },
                ensure_ascii=False,
            ),
            "Quality findings:",
            issues_blob,
            "Draft text:",
            draft_text,
            "Output only the revised full text.",
        ]
    )

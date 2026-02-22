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


# P1: Repair prompts for different issue types

def render_section_repair_prompt(
    plan: GenerationPlan,
    state: GenerationState,
    section_spec: SectionSpec,
    current_text: str,
    quality_issues: list[str],
    retry_index: int,
    original_prompt: str = "",
) -> str:
    """Generate a repair prompt targeting specific quality issues.
    
    Args:
        plan: The generation plan
        state: Current generation state
        section_spec: Section specification
        current_text: The problematic text to fix
        quality_issues: List of identified issues
        retry_index: Current retry attempt (0-based)
        original_prompt: The original generation prompt for context
    """
    # Categorize issues for targeted guidance
    entity_issues = [i for i in quality_issues if "entity" in i.lower() or "missing" in i.lower()]
    length_issues = [i for i in quality_issues if "length" in i.lower()]
    repetition_issues = [i for i in quality_issues if "repetitive" in i.lower() or "similar" in i.lower()]
    other_issues = [i for i in quality_issues if i not in entity_issues + length_issues + repetition_issues]
    
    specific_guidance = []
    
    # P1: Issue-specific repair guidance
    if entity_issues:
        specific_guidance.extend([
            "",
            "ENTITY COVERAGE REQUIREMENTS:",
            "The following REQUIRED entities are missing:",
            *[f"  - {issue}" for issue in entity_issues],
            "",
            "You MUST explicitly mention each missing entity. Strategies:",
            "  - Add a dedicated sentence introducing the entity",
            "  - Integrate naturally into existing content",
            "  - Ensure the entity name matches exactly (case-insensitive)",
        ])
    
    if length_issues:
        specific_guidance.extend([
            "",
            "LENGTH REQUIREMENTS:",
            *[f"  - {issue}" for issue in length_issues],
            "",
            "Strategies to meet length target:",
        ])
        # Check if too short or too long
        if any("too short" in i.lower() or "below" in i.lower() for i in length_issues):
            specific_guidance.extend([
                "  - Expand on key points with more detail",
                "  - Add concrete examples or explanations",
                "  - Elaborate on implications or context",
            ])
        else:
            specific_guidance.extend([
                "  - Remove redundant sentences",
                "  - Condense verbose explanations",
                "  - Focus on core points only",
            ])
    
    if repetition_issues:
        specific_guidance.extend([
            "",
            "REPETITION FIXES:",
            *[f"  - {issue}" for issue in repetition_issues],
            "",
            "Strategies:",
            "  - Use different phrasing and vocabulary",
            "  - Focus on unique aspects for this section",
            "  - Avoid restating concepts from previous sections",
        ])
    
    if other_issues:
        specific_guidance.extend([
            "",
            "OTHER ISSUES TO FIX:",
            *[f"  - {issue}" for issue in other_issues],
        ])
    
    sections = [
        "You are REVISING a previously generated section to fix quality issues.",
        f"Revision attempt: {retry_index + 1}",
        "",
        "=== CURRENT PROBLEMATIC TEXT ===",
        current_text,
        "",
        "=== SECTION REQUIREMENTS ===",
        f"Title: {section_spec.title}",
        f"Target length: ~{section_spec.target_length} tokens (±20% acceptable)",
        "",
        "Key points to cover:",
        *[f"  - {point}" for point in section_spec.key_points],
        "",
        "Required entities (MUST include):",
        *[f"  - {entity}" for entity in section_spec.required_entities],
    ]
    
    if section_spec.constraints:
        sections.extend([
            "",
            "Constraints:",
            *[f"  - {c}" for c in section_spec.constraints],
        ])
    
    sections.extend([
        "",
        "=== ISSUES IDENTIFIED ===",
        *specific_guidance,
        "",
        "=== REVISION REQUIREMENTS ===",
        "1) Fix ALL listed issues above",
        "2) Maintain consistency with already-covered content:",
        f"   Covered key points: {state.covered_key_points}",
        f"   Known entities: {state.known_entities}",
        "3) Preserve the original meaning and structure where possible",
        "4) Output ONLY the revised section text - no explanations, no markdown",
        "",
        "CRITICAL: Your output will be used directly. Do not include:",
        "- Thinking or planning text",
        "- Issue summaries",
        "- Labels like 'Revised text:'",
        "- JSON or code blocks",
        "",
        "Output ONLY the corrected section body text.",
    ])
    
    return "\n".join(sections)

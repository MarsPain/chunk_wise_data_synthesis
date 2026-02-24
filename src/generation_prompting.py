from __future__ import annotations

import json
from typing import Literal

from generation_types import GenerationConfig, GenerationPlan, GenerationState, QualityReport, SectionSpec


PromptLanguage = Literal["en", "zh"]


def _resolve_prompt_language(prompt_language: str) -> PromptLanguage:
    return "zh" if prompt_language == "zh" else "en"


def _none_text(prompt_language: PromptLanguage) -> str:
    return "(无)" if prompt_language == "zh" else "(none)"


def render_plan_prompt(
    topic: str,
    objective: str,
    target_tokens: int,
    audience: str,
    tone: str,
    prompt_language: PromptLanguage = "en",
) -> str:
    language = _resolve_prompt_language(prompt_language)
    audience_text = audience or ("通用技术受众" if language == "zh" else "general technical audience")
    tone_text = tone or ("中性技术风格" if language == "zh" else "neutral technical")
    schema_hint = {
        "topic": topic,
        "objective": objective,
        "audience": audience_text,
        "tone": tone_text,
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

    if language == "zh":
        return "\n\n".join(
            [
                "你正在规划一个长文分节生成任务。",
                "关键要求：只输出下方 JSON 对象，不要输出思考、规划或解释文本。",
                "你的回复必须以 '{' 开始、以 '}' 结束，JSON 前后不能有任何额外文本。",
                "不要使用 markdown 代码块，不要添加注释，只返回原始 JSON。",
                "",
                "请构建完整的生成计划，保证章节衔接合理且覆盖点明确。",
                f"主题：{topic}",
                f"目标：{objective}",
                f"受众：{audience_text}",
                f"语气：{tone_text}",
                f"目标总长度（tokens）：{target_tokens}",
                "",
                "输出结构（只返回符合该结构的 JSON 对象）：",
                json.dumps(schema_hint, ensure_ascii=False),
            ]
        )

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
            f"Audience: {audience_text}",
            f"Tone: {tone_text}",
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
    prompt_language: PromptLanguage = "en",
) -> str:
    language = _resolve_prompt_language(prompt_language)
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

    if language == "zh":
        return "\n\n".join(
            [
                "你正在生成一篇长文中的一个章节。",
                "关键要求：只输出章节正文，不要输出思考、规划或前言。",
                "规则：",
                "1) 严格遵循全局计划和当前章节规格。",
                "2) 术语、实体与时间线需与当前状态保持一致。",
                "3) 避免重复最近文本中已覆盖的要点。",
                "4) 只输出当前章节正文，不要输出 JSON、markdown 或解释。",
                "全局计划：",
                plan_blob,
                "当前状态：",
                state_blob,
                f"最近已生成文本：\n{recent_text or _none_text(language)}",
                "当前章节规格：",
                section_blob,
                "目标长度为近似值（允许 ±20%）。",
            ]
        )

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
            f"Recent generated text:\n{recent_text or _none_text(language)}",
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
    prompt_language: PromptLanguage = "en",
) -> str:
    language = _resolve_prompt_language(prompt_language)
    issues_blob = json.dumps(
        {
            "coverage_missing": quality_report.coverage_missing,
            "terminology_issues": quality_report.terminology_issues,
            "repetition_issues": quality_report.repetition_issues,
            "drift_issues": quality_report.drift_issues,
        },
        ensure_ascii=False,
    )

    if language == "zh":
        return "\n\n".join(
            [
                "你正在对已生成长文执行轻量一致性修订。",
                "关键要求：只输出修订后的完整文本，不要输出思考、规划或前言。",
                "仅允许以下修改：",
                "1) 修正术语一致性",
                "2) 优化章节间衔接",
                "3) 为缺失关键点补充 1-2 句短句",
                "不要进行大幅重写，也不要改变整体结构。",
                "计划：",
                json.dumps(plan.to_dict(), ensure_ascii=False),
                "状态：",
                json.dumps(
                    {
                        "known_entities": state.known_entities,
                        "terminology_map": state.terminology_map,
                        "timeline": state.timeline,
                        "remaining_key_points": state.remaining_key_points,
                    },
                    ensure_ascii=False,
                ),
                "质量发现：",
                issues_blob,
                "草稿文本：",
                draft_text,
                "只输出修订后的完整文本。",
            ]
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


# Prompt compression helpers

def _summarize_covered_points(
    covered: list[str],
    max_items: int = 3,
    prompt_language: PromptLanguage = "en",
) -> str:
    """Summarize covered key points into a short string."""
    language = _resolve_prompt_language(prompt_language)
    if not covered:
        return "暂无" if language == "zh" else "None yet"

    if len(covered) <= max_items:
        return "; ".join(covered)

    recent = covered[-max_items:]
    if language == "zh":
        return f"共 {len(covered)} 个要点，最近：{'; '.join(recent)}"
    return f"{len(covered)} points total, recent: " + "; ".join(recent)


def render_section_prompt_compressed(
    plan: GenerationPlan,
    state: GenerationState,
    section_spec: SectionSpec,
    recent_text: str,
    section_index: int,
    config: GenerationConfig | None = None,
    prompt_language: PromptLanguage = "en",
) -> str:
    """Render a compressed section prompt with minimal context injection."""
    language = _resolve_prompt_language(prompt_language)
    if config is None:
        config = GenerationConfig()

    upcoming_titles = [
        s.title
        for s in plan.sections[section_index + 1:section_index + 1 + config.upcoming_sections_preview]
    ]
    plan_context = {
        "topic": plan.topic,
        "objective": plan.objective,
        "audience": plan.audience,
        "tone": plan.tone,
        "current_section": section_spec.to_dict(),
        "upcoming_sections": upcoming_titles,
        "terminology_preferences": plan.terminology_preferences,
    }

    covered_summary = _summarize_covered_points(
        state.covered_key_points,
        max_items=config.max_covered_points_summary_items,
        prompt_language=language,
    )
    total_points = len(state.covered_key_points) + len(state.remaining_key_points)
    progress = f"{len(state.covered_key_points)}/{total_points}"
    recent_entities = state.known_entities[-config.max_entities_in_prompt:] if state.known_entities else []
    recent_timeline = state.timeline[-config.max_timeline_entries:] if state.timeline else []

    incremental_state = {
        "known_entities": recent_entities,
        "terminology_map": state.terminology_map,
        "timeline": recent_timeline,
        "progress": progress,
        "covered_summary": covered_summary,
        "remaining_points": state.remaining_key_points,
    }

    if language == "zh":
        return "\n\n".join(
            [
                "你正在生成一篇长文中的一个章节。",
                "关键要求：只输出章节正文，不要输出思考、规划或前言。",
                "规则：",
                "1) 严格遵循当前章节规格。",
                "2) 术语需与已知实体保持一致。",
                "3) 覆盖下方列出的全部剩余要点。",
                "4) 不要重复 covered summary 中已覆盖内容。",
                "5) 保持与后续章节衔接。",
                "",
                f"计划上下文：\n{json.dumps(plan_context, ensure_ascii=False, indent=2)}",
                "",
                f"增量状态（进度: {progress}）：\n{json.dumps(incremental_state, ensure_ascii=False, indent=2)}",
                "",
                f"最近已生成文本：\n{recent_text or _none_text(language)}",
                "",
                "只输出当前章节正文。",
            ]
        )

    return "\n\n".join(
        [
            "You are generating one section of a long article.",
            "CRITICAL: Output ONLY the section body text. No thinking, no planning, no preamble.",
            "Rules:",
            "1) Follow the current section spec strictly.",
            "2) Keep terminology consistent with known entities.",
            "3) Cover all remaining points listed below.",
            "4) Do not repeat content summarized in covered summary.",
            "5) Maintain coherence with upcoming sections.",
            "",
            f"Plan context:\n{json.dumps(plan_context, ensure_ascii=False, indent=2)}",
            "",
            f"Incremental state (progress: {progress}):\n{json.dumps(incremental_state, ensure_ascii=False, indent=2)}",
            "",
            f"Recent generated text:\n{recent_text or _none_text(language)}",
            "",
            "Output only the current section body text.",
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
    prompt_language: PromptLanguage = "en",
) -> str:
    """Generate a repair prompt targeting specific quality issues."""
    del plan
    del original_prompt

    language = _resolve_prompt_language(prompt_language)
    entity_issues = [
        i for i in quality_issues if "entity" in i.lower() or "missing" in i.lower() or "实体" in i
    ]
    length_issues = [i for i in quality_issues if "length" in i.lower() or "长度" in i]
    repetition_issues = [
        i for i in quality_issues if "repetitive" in i.lower() or "similar" in i.lower() or "重复" in i
    ]
    other_issues = [i for i in quality_issues if i not in entity_issues + length_issues + repetition_issues]
    specific_guidance: list[str] = []

    if language == "zh":
        if entity_issues:
            specific_guidance.extend(
                [
                    "",
                    "实体覆盖要求：",
                    "以下必需实体缺失：",
                    *[f"  - {issue}" for issue in entity_issues],
                    "",
                    "你必须显式提及每个缺失实体。建议：",
                    "  - 使用专门句子引入该实体",
                    "  - 自然融入现有内容",
                    "  - 确保实体名称匹配（大小写不敏感）",
                ]
            )

        if length_issues:
            specific_guidance.extend(
                [
                    "",
                    "长度要求：",
                    *[f"  - {issue}" for issue in length_issues],
                    "",
                    "满足长度目标的策略：",
                ]
            )
            if any("too short" in i.lower() or "below" in i.lower() or "偏短" in i for i in length_issues):
                specific_guidance.extend(
                    [
                        "  - 展开关键点并补充细节",
                        "  - 添加具体示例或解释",
                        "  - 补充背景或影响",
                    ]
                )
            else:
                specific_guidance.extend(
                    [
                        "  - 删除冗余句子",
                        "  - 压缩冗长解释",
                        "  - 聚焦核心观点",
                    ]
                )

        if repetition_issues:
            specific_guidance.extend(
                [
                    "",
                    "重复性修复：",
                    *[f"  - {issue}" for issue in repetition_issues],
                    "",
                    "策略：",
                    "  - 更换表达与词汇",
                    "  - 聚焦本节独特内容",
                    "  - 避免复述前文概念",
                ]
            )

        if other_issues:
            specific_guidance.extend(
                [
                    "",
                    "其他待修复问题：",
                    *[f"  - {issue}" for issue in other_issues],
                ]
            )

        sections = [
            "你正在修订一个已生成章节以修复质量问题。",
            f"修订轮次：{retry_index + 1}",
            "",
            "=== 当前问题文本 ===",
            current_text,
            "",
            "=== 章节要求 ===",
            f"标题：{section_spec.title}",
            f"目标长度：约 {section_spec.target_length} tokens（允许 ±20%）",
            "",
            "需覆盖的关键要点：",
            *[f"  - {point}" for point in section_spec.key_points],
            "",
            "必需实体（必须包含）：",
            *[f"  - {entity}" for entity in section_spec.required_entities],
        ]
        if section_spec.constraints:
            sections.extend(
                [
                    "",
                    "约束条件：",
                    *[f"  - {c}" for c in section_spec.constraints],
                ]
            )
        sections.extend(
            [
                "",
                "=== 已识别问题 ===",
                *specific_guidance,
                "",
                "=== 修订要求 ===",
                "1) 修复上方列出的全部问题",
                "2) 与已覆盖内容保持一致：",
                f"   已覆盖要点：{state.covered_key_points}",
                f"   已知实体：{state.known_entities}",
                "3) 尽量保留原始含义和结构",
                "4) 只输出修订后的章节正文，不要解释，不要 markdown",
                "",
                "关键要求：输出会被直接使用，请不要包含：",
                "- 思考或规划文本",
                "- 问题总结",
                "- 例如“修订如下：”这类标签",
                "- JSON 或代码块",
                "",
                "只输出修正后的章节正文。",
            ]
        )
        return "\n".join(sections)

    if entity_issues:
        specific_guidance.extend(
            [
                "",
                "ENTITY COVERAGE REQUIREMENTS:",
                "The following REQUIRED entities are missing:",
                *[f"  - {issue}" for issue in entity_issues],
                "",
                "You MUST explicitly mention each missing entity. Strategies:",
                "  - Add a dedicated sentence introducing the entity",
                "  - Integrate naturally into existing content",
                "  - Ensure the entity name matches exactly (case-insensitive)",
            ]
        )

    if length_issues:
        specific_guidance.extend(
            [
                "",
                "LENGTH REQUIREMENTS:",
                *[f"  - {issue}" for issue in length_issues],
                "",
                "Strategies to meet length target:",
            ]
        )
        if any("too short" in i.lower() or "below" in i.lower() for i in length_issues):
            specific_guidance.extend(
                [
                    "  - Expand on key points with more detail",
                    "  - Add concrete examples or explanations",
                    "  - Elaborate on implications or context",
                ]
            )
        else:
            specific_guidance.extend(
                [
                    "  - Remove redundant sentences",
                    "  - Condense verbose explanations",
                    "  - Focus on core points only",
                ]
            )

    if repetition_issues:
        specific_guidance.extend(
            [
                "",
                "REPETITION FIXES:",
                *[f"  - {issue}" for issue in repetition_issues],
                "",
                "Strategies:",
                "  - Use different phrasing and vocabulary",
                "  - Focus on unique aspects for this section",
                "  - Avoid restating concepts from previous sections",
            ]
        )

    if other_issues:
        specific_guidance.extend(
            [
                "",
                "OTHER ISSUES TO FIX:",
                *[f"  - {issue}" for issue in other_issues],
            ]
        )

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
        sections.extend(
            [
                "",
                "Constraints:",
                *[f"  - {c}" for c in section_spec.constraints],
            ]
        )
    sections.extend(
        [
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
        ]
    )
    return "\n".join(sections)

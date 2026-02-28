import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from prompts.generation import (
    render_consistency_prompt,
    render_plan_prompt,
    render_section_prompt,
    render_section_prompt_compressed,
    render_section_repair_prompt,
)
from generation_types import (
    GenerationConfig,
    GenerationPlan,
    GenerationState,
    QualityReport,
    SectionSpec,
)


def _build_plan() -> GenerationPlan:
    return GenerationPlan(
        topic="示例主题",
        objective="示例目标",
        audience="工程师",
        tone="技术中性",
        target_total_length=300,
        narrative_voice="third-person",
        do_not_include=["unsupported claims"],
        terminology_preferences={"术语A": "术语A"},
        sections=[
            SectionSpec(
                title="第一节",
                key_points=["要点1"],
                required_entities=["实体A"],
                constraints=[],
                target_length=100,
            )
        ],
    )


def _build_state() -> GenerationState:
    return GenerationState(
        known_entities=["实体A"],
        terminology_map={"术语A": "术语A"},
        timeline=["2024"],
        covered_key_points=[],
        remaining_key_points=["要点1"],
    )


class GenerationPromptLanguageTests(unittest.TestCase):
    def test_plan_prompt_supports_chinese(self) -> None:
        prompt = render_plan_prompt(
            topic="示例主题",
            objective="示例目标",
            target_tokens=300,
            audience="工程师",
            tone="技术中性",
            prompt_language="zh",
        )
        self.assertIn("你正在规划一个长文分节生成任务。", prompt)

    def test_section_prompt_supports_chinese(self) -> None:
        prompt = render_section_prompt(
            plan=_build_plan(),
            state=_build_state(),
            recent_text="",
            section_spec=_build_plan().sections[0],
            prompt_language="zh",
        )
        self.assertIn("你正在生成一篇长文中的一个章节。", prompt)

    def test_consistency_prompt_supports_chinese(self) -> None:
        prompt = render_consistency_prompt(
            plan=_build_plan(),
            state=_build_state(),
            draft_text="草稿",
            quality_report=QualityReport(),
            prompt_language="zh",
        )
        self.assertIn("你正在对已生成长文执行轻量一致性修订。", prompt)

    def test_compressed_prompt_supports_chinese(self) -> None:
        prompt = render_section_prompt_compressed(
            plan=_build_plan(),
            state=_build_state(),
            section_spec=_build_plan().sections[0],
            recent_text="",
            section_index=0,
            config=GenerationConfig(),
            prompt_language="zh",
        )
        self.assertIn("你正在生成一篇长文中的一个章节。", prompt)

    def test_repair_prompt_supports_chinese(self) -> None:
        prompt = render_section_repair_prompt(
            plan=_build_plan(),
            state=_build_state(),
            section_spec=_build_plan().sections[0],
            current_text="当前文本",
            quality_issues=["Missing required entity: '实体A'"],
            retry_index=1,
            prompt_language="zh",
        )
        self.assertIn("你正在修订一个已生成章节以修复质量问题。", prompt)


if __name__ == "__main__":
    unittest.main()

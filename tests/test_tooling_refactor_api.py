import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generation_types import GenerationPlan, GenerationState, QualityReport, SectionSpec


def _build_plan() -> GenerationPlan:
    return GenerationPlan(
        topic="Example topic",
        objective="Example objective",
        audience="Engineers",
        tone="Neutral",
        target_total_length=300,
        narrative_voice="third-person",
        do_not_include=["unsupported claims"],
        terminology_preferences={"term": "term"},
        sections=[
            SectionSpec(
                title="Section A",
                key_points=["point A"],
                required_entities=["entity A"],
                constraints=[],
                target_length=120,
            )
        ],
    )


def _build_state() -> GenerationState:
    return GenerationState(
        known_entities=["entity A"],
        terminology_map={"term": "term"},
        timeline=["2024"],
        covered_key_points=[],
        remaining_key_points=["point A"],
    )


class TokenizationApiTests(unittest.TestCase):
    def test_tokenization_exports_required_symbols(self) -> None:
        from tokenization import Tokenizer, WhitespaceTokenizer, take_last_tokens

        tokenizer: Tokenizer = WhitespaceTokenizer()
        self.assertEqual(take_last_tokens("a b c d", tokenizer, 2), "c d")


class PromptsApiTests(unittest.TestCase):
    def test_prompts_base_helpers(self) -> None:
        from prompts.base import _none_text, _resolve_prompt_language

        self.assertEqual(_resolve_prompt_language("zh"), "zh")
        self.assertEqual(_resolve_prompt_language("anything"), "en")
        self.assertEqual(_none_text("en"), "(none)")
        self.assertEqual(_none_text("zh"), "(无)")

    def test_rephrase_prompt_api(self) -> None:
        from prompts.rephrase import RewriteRequest, render_rewrite_prompt

        prompt = render_rewrite_prompt(
            RewriteRequest(
                style_instruction="neutral",
                global_anchor="anchor",
                generated_prefix="prefix",
                current_chunk="chunk",
                retry_index=0,
                strict_fidelity=False,
                prompt_language="en",
            )
        )
        self.assertIn("You are a faithful rewriter.", prompt)

    def test_generation_prompt_api(self) -> None:
        from prompts.generation import (
            _summarize_covered_points,
            render_consistency_prompt,
            render_plan_prompt,
            render_section_prompt,
            render_section_prompt_compressed,
            render_section_repair_prompt,
        )

        plan = _build_plan()
        state = _build_state()
        section = plan.sections[0]

        self.assertIn(
            "You are planning a long-form, section-wise generation task.",
            render_plan_prompt(
                topic=plan.topic,
                objective=plan.objective,
                target_tokens=plan.target_total_length,
                audience=plan.audience,
                tone=plan.tone,
            ),
        )
        self.assertIn(
            "You are generating one section of a long article.",
            render_section_prompt(plan, state, "", section),
        )
        self.assertIn(
            "You are running a light consistency pass",
            render_consistency_prompt(plan, state, "draft", QualityReport()),
        )
        self.assertIn(
            "You are generating one section of a long article.",
            render_section_prompt_compressed(
                plan=plan,
                state=state,
                section_spec=section,
                recent_text="",
                section_index=0,
            ),
        )
        self.assertIn(
            "REVISING a previously generated section",
            render_section_repair_prompt(
                plan=plan,
                state=state,
                section_spec=section,
                current_text="text",
                quality_issues=[],
                retry_index=0,
            ),
        )
        self.assertEqual(_summarize_covered_points([]), "None yet")

    def test_prompts_package_exports_expected_symbols(self) -> None:
        from prompts import RewriteRequest, render_plan_prompt
        from prompts.generation import render_plan_prompt as render_plan_prompt_impl
        from prompts.rephrase import RewriteRequest as RewriteRequestImpl

        self.assertIs(RewriteRequest, RewriteRequestImpl)
        self.assertIs(render_plan_prompt, render_plan_prompt_impl)


if __name__ == "__main__":
    unittest.main()

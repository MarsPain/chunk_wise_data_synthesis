import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generation_types import GenerationPlan, SectionSpec
from quality.evaluation import (
    CoherenceMetrics,
    compare_chunked_vs_one_shot,
    evaluate_generation_coherence,
)


def _build_plan() -> GenerationPlan:
    return GenerationPlan(
        topic="Chunk-wise generation",
        objective="Explain how to write coherent long-form technical articles",
        audience="ML engineers",
        tone="neutral technical",
        target_total_length=800,
        sections=[
            SectionSpec(
                title="Problem",
                key_points=["define coherence problem", "state section boundary risks"],
                required_entities=["coherence", "section boundary"],
                constraints=[],
                target_length=250,
            ),
            SectionSpec(
                title="Method",
                key_points=["introduce transition contract", "show quality loop"],
                required_entities=["transition contract", "quality loop"],
                constraints=[],
                target_length=300,
            ),
        ],
        terminology_preferences={"LLM": "large language model"},
        narrative_voice="third-person",
        do_not_include=["fiction"],
    )


class GenerationEvaluationTests(unittest.TestCase):
    def test_coherent_output_scores_higher_than_repetitive_output(self) -> None:
        plan = _build_plan()
        coherent_sections = [
            (
                "The coherence problem appears when each section is generated independently. "
                "A section boundary can break narrative flow if the topic handoff is missing."
            ),
            (
                "Building on that boundary risk, the method applies a transition contract. "
                "The quality loop then checks missing points and repairs section links."
            ),
        ]
        repetitive_sections = [
            "Chunk-wise generation uses chunk-wise generation and repeats chunk-wise generation.",
            "Chunk-wise generation uses chunk-wise generation and repeats chunk-wise generation.",
        ]

        coherent_metrics = evaluate_generation_coherence(plan=plan, section_outputs=coherent_sections)
        repetitive_metrics = evaluate_generation_coherence(plan=plan, section_outputs=repetitive_sections)

        self.assertGreater(coherent_metrics.aggregate_score, repetitive_metrics.aggregate_score)
        self.assertLess(coherent_metrics.repetition_risk, repetitive_metrics.repetition_risk)

    def test_drift_and_terminology_issues_are_reported(self) -> None:
        plan = _build_plan()
        drifting_sections = [
            "This section discusses tomatoes, basil, and oven timing for pasta.",
            "The LLM can be optimized with prompts, but the phrase is not normalized.",
        ]

        metrics = evaluate_generation_coherence(plan=plan, section_outputs=drifting_sections)

        self.assertGreater(metrics.drift_risk, 0.4)
        self.assertGreaterEqual(metrics.terminology_issue_count, 1)

    def test_chinese_text_is_supported_by_metrics(self) -> None:
        plan = GenerationPlan(
            topic="分章节长文生成",
            objective="提升整篇连贯性",
            audience="工程师",
            tone="技术中性",
            target_total_length=600,
            sections=[
                SectionSpec(
                    title="现状问题",
                    key_points=["边界断裂", "术语不一致"],
                    required_entities=["边界", "术语"],
                    constraints=[],
                    target_length=200,
                ),
                SectionSpec(
                    title="优化方案",
                    key_points=["过渡句契约", "回修闭环"],
                    required_entities=["过渡句", "回修"],
                    constraints=[],
                    target_length=250,
                ),
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )
        sections = [
            "当前方案在章节边界处容易断裂，术语也会漂移。",
            "接下来通过过渡句契约和回修闭环，提升整篇连贯性。",
        ]

        metrics = evaluate_generation_coherence(plan=plan, section_outputs=sections)

        self.assertGreater(metrics.boundary_coherence, 0.0)
        self.assertLess(metrics.drift_risk, 1.0)

    def test_compare_chunked_vs_one_shot_reports_winner(self) -> None:
        chunked = CoherenceMetrics(
            boundary_coherence=0.7,
            repetition_risk=0.2,
            drift_risk=0.1,
            terminology_issue_count=0,
            coverage_missing_count=0,
            aggregate_score=0.85,
        )
        one_shot = CoherenceMetrics(
            boundary_coherence=0.5,
            repetition_risk=0.3,
            drift_risk=0.2,
            terminology_issue_count=1,
            coverage_missing_count=1,
            aggregate_score=0.62,
        )

        comparison = compare_chunked_vs_one_shot(
            case_id="case-1",
            chunked_metrics=chunked,
            one_shot_metrics=one_shot,
        )

        self.assertEqual(comparison.winner, "chunked")
        self.assertGreater(comparison.score_delta, 0.0)


if __name__ == "__main__":
    unittest.main()

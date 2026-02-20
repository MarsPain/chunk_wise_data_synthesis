import unittest

from path_setup import ensure_src_path

ensure_src_path()

from generation_quality import (
    OutlineCoverageChecker,
    RepetitionAndDriftChecker,
    StrictConsistencyEditGuard,
    TerminologyConsistencyChecker,
)
from generation_types import GenerationPlan, SectionSpec


class GenerationQualityTests(unittest.TestCase):
    def test_outline_coverage_checker_reports_missing_points(self) -> None:
        plan = GenerationPlan(
            topic="Chunk-wise generation",
            objective="teach",
            audience="engineers",
            tone="neutral",
            target_total_length=400,
            sections=[
                SectionSpec(
                    title="Intro",
                    key_points=["define chunk-wise AR", "state table tracks entities"],
                    required_entities=[],
                    constraints=[],
                    target_length=120,
                )
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = OutlineCoverageChecker()
        missing = checker.find_missing(plan=plan, text="This section only define chunk-wise AR clearly.")

        self.assertIn("state table tracks entities", missing)
        self.assertNotIn("define chunk-wise AR", missing)

    def test_terminology_checker_flags_nonpreferred_term(self) -> None:
        plan = GenerationPlan(
            topic="Optimization",
            objective="explain",
            audience="beginners",
            tone="neutral",
            target_total_length=300,
            sections=[],
            terminology_preferences={"梯度累积": "gradient accumulation"},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = TerminologyConsistencyChecker()
        issues = checker.find_issues(plan=plan, text="梯度累积 can reduce memory pressure.")

        self.assertEqual(len(issues), 1)
        self.assertIn("gradient accumulation", issues[0])

    def test_repetition_and_drift_checker_detects_both(self) -> None:
        plan = GenerationPlan(
            topic="Transformer training",
            objective="explain",
            audience="engineers",
            tone="neutral",
            target_total_length=600,
            sections=[
                SectionSpec(
                    title="Chunking",
                    key_points=["chunk boundaries"],
                    required_entities=["Transformer"],
                    constraints=[],
                    target_length=200,
                ),
                SectionSpec(
                    title="State",
                    key_points=["state updates"],
                    required_entities=["state table"],
                    constraints=[],
                    target_length=200,
                ),
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = RepetitionAndDriftChecker(repetition_threshold=0.8, drift_overlap_threshold=0.1)
        section_outputs = [
            "Transformer chunk boundaries improve coherence and chunk boundaries reduce drift.",
            "Transformer chunk boundaries improve coherence and chunk boundaries reduce drift.",
            "This recipe uses tomatoes basil and oven timing for pasta dinner.",
        ]

        repetition, drift = checker.find_issues(plan=plan, section_outputs=section_outputs)

        self.assertEqual(len(repetition), 1)
        self.assertEqual(len(drift), 1)

    def test_strict_consistency_edit_guard_rejects_large_rewrite(self) -> None:
        guard = StrictConsistencyEditGuard(min_token_jaccard=0.7, max_length_ratio=1.3)
        original_text = "Chunk-wise generation keeps structure and tracks entities across sections."
        rewritten_text = "The history of cooking focuses on ingredients, ovens, and desserts in Europe."

        accepted_text, used_fallback = guard.apply(original_text=original_text, candidate_text=rewritten_text)

        self.assertTrue(used_fallback)
        self.assertEqual(accepted_text, original_text)


if __name__ == "__main__":
    unittest.main()

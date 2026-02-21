import unittest

from path_setup import ensure_src_path

ensure_src_path()

from generation_quality import (
    EntityPresenceChecker,
    NumericFactChecker,
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


class EntityPresenceCheckerTests(unittest.TestCase):
    def test_detects_missing_entity_in_section(self) -> None:
        plan = GenerationPlan(
            topic="AI Training",
            objective="explain",
            audience="engineers",
            tone="technical",
            target_total_length=500,
            sections=[
                SectionSpec(
                    title="Basics",
                    key_points=["intro to training"],
                    required_entities=["Transformer", "GPU cluster", "training data"],
                    constraints=[],
                    target_length=200,
                )
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = EntityPresenceChecker()
        # Missing "GPU cluster" and "training data"
        section_outputs = ["This section covers Transformer architecture in detail."]

        missing = checker.find_missing(plan=plan, section_outputs=section_outputs)

        self.assertEqual(len(missing), 2)
        self.assertTrue(any("GPU cluster" in m for m in missing))
        self.assertTrue(any("training data" in m for m in missing))

    def test_recognizes_entity_with_hyphen_variant(self) -> None:
        plan = GenerationPlan(
            topic="Chunking",
            objective="explain",
            audience="engineers",
            tone="neutral",
            target_total_length=300,
            sections=[
                SectionSpec(
                    title="Method",
                    key_points=["how to chunk"],
                    required_entities=["state table"],
                    constraints=[],
                    target_length=150,
                )
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = EntityPresenceChecker()
        # Uses hyphen variant "state-table" instead of "state table"
        section_outputs = ["The state-table tracks entities across chunks."]

        missing = checker.find_missing(plan=plan, section_outputs=section_outputs)

        self.assertEqual(len(missing), 0)

    def test_multi_section_entity_checking(self) -> None:
        plan = GenerationPlan(
            topic="System Design",
            objective="document",
            audience="architects",
            tone="formal",
            target_total_length=600,
            sections=[
                SectionSpec(
                    title="Frontend",
                    key_points=["ui design"],
                    required_entities=["React", "TypeScript"],
                    constraints=[],
                    target_length=200,
                ),
                SectionSpec(
                    title="Backend",
                    key_points=["api design"],
                    required_entities=["FastAPI", "PostgreSQL"],
                    constraints=[],
                    target_length=200,
                ),
            ],
            terminology_preferences={},
            narrative_voice="third-person",
            do_not_include=[],
        )

        checker = EntityPresenceChecker()
        # First section missing TypeScript, second section complete
        section_outputs = [
            "We use React for the frontend.",  # Missing TypeScript
            "Backend uses FastAPI with PostgreSQL database.",  # Complete
        ]

        missing = checker.find_missing(plan=plan, section_outputs=section_outputs)

        self.assertEqual(len(missing), 1)
        self.assertIn("Section 1", missing[0])
        self.assertIn("TypeScript", missing[0])


class NumericFactCheckerTests(unittest.TestCase):
    def test_detects_missing_year(self) -> None:
        checker = NumericFactChecker()
        source = "The Transformer architecture was introduced in 2017 by Google."
        generated = "The Transformer architecture was introduced by Google."  # Missing 2017

        issues = checker.find_missing(source, generated)

        self.assertEqual(len(issues), 1)
        self.assertIn("2017", issues[0])
        self.assertIn("year", issues[0].lower())

    def test_detects_missing_percentage(self) -> None:
        checker = NumericFactChecker()
        source = "The model achieved 95.5% accuracy on the test set."
        generated = "The model achieved high accuracy on the test set."  # Missing 95.5%

        issues = checker.find_missing(source, generated)

        self.assertEqual(len(issues), 1)
        self.assertIn("95.5%", issues[0])

    def test_detects_missing_quantity(self) -> None:
        checker = NumericFactChecker()
        source = "The dataset contains 1.5 million training examples."
        generated = "The dataset contains many training examples."  # Missing 1.5 million

        issues = checker.find_missing(source, generated)

        self.assertEqual(len(issues), 1)
        self.assertIn("1.5 million", issues[0])

    def test_passes_when_all_facts_present(self) -> None:
        checker = NumericFactChecker()
        source = "Released in 2020, version 2.0 improved performance by 15%."
        generated = "The 2020 release of version 2.0 brought 15% improvement."

        issues = checker.find_missing(source, generated)

        self.assertEqual(len(issues), 0)

    def test_detects_multiple_missing_facts(self) -> None:
        checker = NumericFactChecker()
        source = "In 2019, 2020, and 2021, the company grew by 25%, 30%, and 35%."
        generated = "The company experienced growth in those years."  # Missing all numbers

        issues = checker.find_missing(source, generated)

        # Should detect missing years (3) and percentages (3)
        self.assertGreaterEqual(len(issues), 4)

    def test_handles_percentage_word_form(self) -> None:
        checker = NumericFactChecker()
        source = "The success rate is 95 percent."
        generated = "The success rate is high."  # Missing 95 percent

        issues = checker.find_missing(source, generated)

        self.assertEqual(len(issues), 1)


if __name__ == "__main__":
    unittest.main()

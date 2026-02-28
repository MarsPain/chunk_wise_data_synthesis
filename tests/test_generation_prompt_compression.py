"""Tests for prompt compression and incremental state optimization."""

import unittest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generation_types import GenerationConfig, GenerationPlan, GenerationState, SectionSpec
from prompts.generation import (
    render_section_prompt,
    render_section_prompt_compressed,
    _summarize_covered_points,
)


class SummarizeCoveredPointsTests(unittest.TestCase):
    """Test the _summarize_covered_points helper function."""

    def test_empty_list_returns_none_yet(self) -> None:
        result = _summarize_covered_points([])
        self.assertEqual(result, "None yet")

    def test_single_point(self) -> None:
        result = _summarize_covered_points(["Background introduction"])
        self.assertEqual(result, "Background introduction")

    def test_few_points_all_shown(self) -> None:
        points = ["Point A", "Point B", "Point C"]
        result = _summarize_covered_points(points, max_items=3)
        self.assertEqual(result, "Point A; Point B; Point C")

    def test_many_points_summarized(self) -> None:
        points = [f"Point {i}" for i in range(10)]
        result = _summarize_covered_points(points, max_items=3)
        # Should show total count and recent 3
        self.assertIn("10 points total", result)
        self.assertIn("Point 7", result)
        self.assertIn("Point 8", result)
        self.assertIn("Point 9", result)
        # Should not show early points
        self.assertNotIn("Point 0", result)
        self.assertNotIn("Point 5", result)

    def test_custom_max_items(self) -> None:
        points = [f"Point {i}" for i in range(5)]
        result = _summarize_covered_points(points, max_items=2)
        self.assertIn("5 points total", result)
        self.assertIn("Point 3", result)
        self.assertIn("Point 4", result)


class CompressedPromptTests(unittest.TestCase):
    """Test the compressed section prompt generation."""

    def _create_test_plan(self) -> GenerationPlan:
        """Create a test plan with multiple sections."""
        return GenerationPlan(
            topic="AI Technology",
            objective="Explain transformer architecture",
            audience="technical",
            tone="neutral",
            target_total_length=2000,
            narrative_voice="third-person",
            do_not_include=["unsupported claims"],
            terminology_preferences={"AI": "artificial intelligence"},
            sections=[
                SectionSpec(
                    title="Introduction",
                    key_points=["Background", "Problem statement"],
                    required_entities=["Transformer"],
                    constraints=["Keep it simple"],
                    target_length=300,
                ),
                SectionSpec(
                    title="Architecture",
                    key_points=["Attention mechanism", "Multi-head attention", "Feed-forward layers"],
                    required_entities=["Attention", "FFN"],
                    constraints=[],
                    target_length=500,
                ),
                SectionSpec(
                    title="Applications",
                    key_points=["NLP tasks", "Vision tasks"],
                    required_entities=["BERT", "GPT"],
                    constraints=[],
                    target_length=400,
                ),
            ],
        )

    def _create_test_state(self, covered: list[str], remaining: list[str]) -> GenerationState:
        """Create a test state with specified covered/remaining points."""
        return GenerationState(
            known_entities=["Transformer", "BERT", "GPT", "Attention"],
            terminology_map={"AI": "artificial intelligence"},
            timeline=["2017", "2018"],
            covered_key_points=covered,
            remaining_key_points=remaining,
        )

    def test_compressed_prompt_excludes_other_sections_details(self) -> None:
        """Compressed prompt should not include full details of other sections."""
        plan = self._create_test_plan()
        state = self._create_test_state([], plan.sections[0].key_points)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
        )

        # Should contain current section details
        self.assertIn("Introduction", prompt)
        self.assertIn("Background", prompt)

        # Should NOT contain other sections' key_points (compression benefit)
        self.assertNotIn("Multi-head attention", prompt)
        self.assertNotIn("Feed-forward layers", prompt)
        self.assertNotIn("Vision tasks", prompt)

    def test_compressed_prompt_includes_upcoming_section_titles(self) -> None:
        """Compressed prompt should include upcoming section titles for context."""
        plan = self._create_test_plan()
        state = self._create_test_state([], plan.sections[0].key_points)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
        )

        # Should include upcoming section titles
        self.assertIn("Architecture", prompt)
        self.assertIn("Applications", prompt)

    def test_compressed_prompt_uses_summary_for_covered_points(self) -> None:
        """Compressed prompt should summarize covered points, not list them all."""
        plan = self._create_test_plan()
        # Simulate progress: first 5 sections done (hypothetically)
        covered = [f"Covered point {i}" for i in range(10)]
        remaining = ["Remaining point"]
        state = self._create_test_state(covered, remaining)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
        )

        # Should show summary, not full list
        self.assertIn("10 points total", prompt)
        # Should NOT contain all individual covered points
        self.assertNotIn("Covered point 0", prompt)
        self.assertNotIn("Covered point 5", prompt)

    def test_compressed_prompt_includes_only_remaining_points(self) -> None:
        """Compressed prompt should include only remaining key points."""
        plan = self._create_test_plan()
        covered = ["Already covered"]
        remaining = ["Still need this", "And this too"]
        state = self._create_test_state(covered, remaining)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
        )

        # Should include remaining points
        self.assertIn("Still need this", prompt)
        self.assertIn("And this too", prompt)

    def test_compressed_prompt_has_progress_indicator(self) -> None:
        """Compressed prompt should include progress indicator."""
        plan = self._create_test_plan()
        covered = ["Point 1", "Point 2", "Point 3"]
        remaining = ["Point 4", "Point 5"]
        state = self._create_test_state(covered, remaining)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
        )

        # Should show progress like "3/5"
        self.assertIn("3/5", prompt)

    def test_compressed_prompt_limits_entities(self) -> None:
        """Compressed prompt should limit number of entities shown."""
        plan = self._create_test_plan()
        many_entities = [f"Entity{i}" for i in range(50)]
        state = GenerationState(
            known_entities=many_entities,
            terminology_map={},
            timeline=[],
            covered_key_points=[],
            remaining_key_points=["Point"],
        )
        current_section = plan.sections[0]

        config = GenerationConfig(max_entities_in_prompt=10)
        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
            config=config,
        )

        # First entity should be present (in the sliced portion)
        self.assertIn("Entity40", prompt)  # Last 10: Entity40-49
        # Early entity should NOT be present
        self.assertNotIn("Entity0", prompt)
        self.assertNotIn("Entity30", prompt)

    def test_compressed_prompt_limits_timeline(self) -> None:
        """Compressed prompt should limit timeline entries."""
        plan = self._create_test_plan()
        many_years = [str(2000 + i) for i in range(20)]
        state = GenerationState(
            known_entities=[],
            terminology_map={},
            timeline=many_years,
            covered_key_points=[],
            remaining_key_points=["Point"],
        )
        current_section = plan.sections[0]

        config = GenerationConfig(max_timeline_entries=5)
        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="",
            section_index=0,
            config=config,
        )

        # Recent years should be present
        self.assertIn("2019", prompt)
        # Old years should NOT be present
        self.assertNotIn("2000", prompt)
        self.assertNotIn("2010", prompt)

    def test_compressed_prompt_is_shorter_than_original(self) -> None:
        """Compressed prompt should be significantly shorter than original."""
        plan = self._create_test_plan()
        # Simulate a long document state
        covered = [f"Covered point {i}" for i in range(50)]
        remaining = [f"Remaining point {i}" for i in range(10)]
        state = self._create_test_state(covered, remaining)
        current_section = plan.sections[0]

        original_prompt = render_section_prompt(
            plan=plan,
            state=state,
            recent_text="Some recent text for context",
            section_spec=current_section,
        )

        compressed_prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="Some recent text for context",
            section_index=0,
        )

        # Compressed should be significantly shorter
        original_len = len(original_prompt)
        compressed_len = len(compressed_prompt)
        reduction_ratio = compressed_len / original_len

        self.assertLess(
            reduction_ratio,
            0.7,  # Expect at least 30% reduction
            f"Compressed prompt ({compressed_len}) should be at least 30% shorter "
            f"than original ({original_len}), but only got {reduction_ratio:.1%} reduction"
        )

    def test_compressed_prompt_includes_essential_info(self) -> None:
        """Compressed prompt should still include essential generation info."""
        plan = self._create_test_plan()
        state = self._create_test_state([], plan.sections[0].key_points)
        current_section = plan.sections[0]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=current_section,
            recent_text="Recent context",
            section_index=0,
        )

        # Essential info should still be present
        self.assertIn(plan.topic, prompt)
        self.assertIn(plan.objective, prompt)
        self.assertIn(current_section.title, prompt)
        self.assertIn("Recent context", prompt)
        # Required entities should be present
        for entity in current_section.required_entities:
            self.assertIn(entity, prompt)

    def test_last_section_has_no_upcoming_sections(self) -> None:
        """When generating the last section, upcoming sections should be empty."""
        plan = self._create_test_plan()
        state = self._create_test_state([], [])
        last_section = plan.sections[-1]

        prompt = render_section_prompt_compressed(
            plan=plan,
            state=state,
            section_spec=last_section,
            recent_text="",
            section_index=len(plan.sections) - 1,
        )

        # Should not crash and should contain the section
        self.assertIn(last_section.title, prompt)


class GenerationConfigCompressionTests(unittest.TestCase):
    """Test GenerationConfig compression settings."""

    def test_default_compression_settings(self) -> None:
        """Default config should have sensible compression defaults."""
        config = GenerationConfig()

        # Check compression-related defaults
        self.assertEqual(config.prompt_compression_enabled, True)
        self.assertEqual(config.max_covered_points_summary_items, 3)
        self.assertEqual(config.max_entities_in_prompt, 20)
        self.assertEqual(config.max_timeline_entries, 5)
        self.assertEqual(config.upcoming_sections_preview, 2)


if __name__ == "__main__":
    unittest.main()

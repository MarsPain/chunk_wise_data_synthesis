import json
import unittest
from dataclasses import dataclass
from typing import List

from path_setup import ensure_src_path

ensure_src_path()

from pipelines import ChunkWiseGenerationPipeline
from generation_types import GenerationConfig, GenerationPlan, SectionSpec
from model import LLMRequest
from tokenization import WhitespaceTokenizer


@dataclass
class _RecordedCall:
    request: LLMRequest


class ScriptedLLMModel:
    def __init__(self, scripted_outputs: List[str]) -> None:
        self._scripted_outputs = scripted_outputs
        self._cursor = 0
        self.calls: List[_RecordedCall] = []

    def generate(self, request: LLMRequest) -> str:
        self.calls.append(_RecordedCall(request=request))
        if self._cursor >= len(self._scripted_outputs):
            raise RuntimeError("scripted outputs exhausted")
        value = self._scripted_outputs[self._cursor]
        self._cursor += 1
        return value


def _build_manual_plan() -> GenerationPlan:
    return GenerationPlan(
        topic="Chunk-wise generation",
        objective="teach",
        audience="ML engineers",
        tone="neutral technical",
        target_total_length=500,
        sections=[
            SectionSpec(
                title="Plan",
                key_points=["global anchor sets scope"],
                required_entities=["global anchor"],
                constraints=["no marketing"],
                target_length=120,
            ),
            SectionSpec(
                title="State",
                key_points=["state table tracks entities"],
                required_entities=["state table"],
                constraints=["no repetition"],
                target_length=120,
            ),
        ],
        terminology_preferences={"global anchor": "global anchor"},
        narrative_voice="third-person",
        do_not_include=["fiction"],
    )


class GenerationPipelineTests(unittest.TestCase):
    def test_manual_plan_runs_section_generation_and_consistency(self) -> None:
        plan = _build_manual_plan()
        model = ScriptedLLMModel(
            scripted_outputs=[
                "The global anchor sets scope and keeps constraints explicit.",
                "The state table tracks entities and marks covered points.",
                "The global anchor sets scope and keeps constraints explicit. The state table tracks entities and marks covered points.",
            ]
        )
        pipeline = ChunkWiseGenerationPipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=GenerationConfig(prefix_window_tokens=20),
        )

        result = pipeline.run(manual_plan=plan)

        self.assertEqual([call.request.task for call in model.calls], ["section_generation", "section_generation", "consistency_pass"])
        self.assertEqual(len(result.section_outputs), 2)
        self.assertEqual(result.final_state.remaining_key_points, [])
        self.assertTrue(result.qc_report.consistency_pass_applied)

    def test_auto_plan_runs_plan_then_sections_then_consistency(self) -> None:
        model = ScriptedLLMModel(
            scripted_outputs=[
                json.dumps(
                    {
                        "topic": "Chunk-wise generation",
                        "objective": "teach",
                        "audience": "ML engineers",
                        "tone": "neutral",
                        "target_total_length": 300,
                        "narrative_voice": "third-person",
                        "do_not_include": ["fiction"],
                        "terminology_preferences": {"state table": "state table"},
                        "sections": [
                            {
                                "title": "Intro",
                                "key_points": ["global anchor sets scope"],
                                "required_entities": ["global anchor"],
                                "constraints": [],
                                "target_length": 120,
                            },
                            {
                                "title": "State",
                                "key_points": ["state table tracks entities"],
                                "required_entities": ["state table"],
                                "constraints": [],
                                "target_length": 120,
                            },
                        ],
                    }
                ),
                "A global anchor sets scope for long generation.",
                "The state table tracks entities and remaining points.",
                "A global anchor sets scope for long generation. The state table tracks entities and remaining points.",
            ]
        )
        pipeline = ChunkWiseGenerationPipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=GenerationConfig(prefix_window_tokens=16),
        )

        _ = pipeline.run(
            topic="Chunk-wise generation",
            objective="teach",
            target_tokens=300,
            audience="ML engineers",
            tone="neutral",
        )

        self.assertEqual(
            [call.request.task for call in model.calls],
            ["plan_generation", "section_generation", "section_generation", "consistency_pass"],
        )

    def test_rolling_window_is_applied_to_next_section_prompt(self) -> None:
        plan = _build_manual_plan()
        model = ScriptedLLMModel(
            scripted_outputs=[
                # Section 1 first attempt (missing 'global anchor', triggers retry)
                "t0 t1 t2 t3 t4 t5 t6 t7",
                # Section 1 repair attempt (still missing 'global anchor')
                "t0 t1 t2 t3 t4 t5 t6 t7",
                # Section 2 (contains 'state table')
                "state table tracks entities",
                # Consistency pass
                "t0 t1 t2 t3 t4 t5 t6 t7 state table tracks entities",
            ]
        )
        pipeline = ChunkWiseGenerationPipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=GenerationConfig(prefix_window_tokens=4),
        )

        _ = pipeline.run(manual_plan=plan)

        second_prompt = model.calls[2].request.prompt  # Section 2 call (after retry)
        self.assertIn("t4 t5 t6 t7", second_prompt)
        self.assertNotIn("t0 t1", second_prompt)

    def test_consistency_guard_blocks_large_rewrite(self) -> None:
        plan = _build_manual_plan()
        model = ScriptedLLMModel(
            scripted_outputs=[
                "The global anchor sets scope and keeps constraints explicit.",
                "The state table tracks entities and marks covered points.",
                "This recipe discusses tomatoes and basil unrelated to generation.",
            ]
        )
        pipeline = ChunkWiseGenerationPipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=GenerationConfig(prefix_window_tokens=20),
        )

        result = pipeline.run(manual_plan=plan)

        self.assertTrue(result.qc_report.consistency_pass_used_fallback)
        self.assertIn("global anchor", result.final_text)
        self.assertIn("state table", result.final_text)

    def test_prompt_language_zh_is_used_in_section_prompt(self) -> None:
        plan = _build_manual_plan()
        model = ScriptedLLMModel(
            scripted_outputs=[
                "global anchor appears in this section output.",
                "state table appears in this section output.",
                "global anchor appears in this section output. state table appears in this section output.",
            ]
        )
        pipeline = ChunkWiseGenerationPipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=GenerationConfig(
                prefix_window_tokens=20,
                prompt_language="zh",
            ),
        )

        _ = pipeline.run(manual_plan=plan)

        first_section_prompt = model.calls[0].request.prompt
        self.assertIn("你正在生成一篇长文中的一个章节。", first_section_prompt)


if __name__ == "__main__":
    unittest.main()

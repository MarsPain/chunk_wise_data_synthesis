import os
from pathlib import Path
import unittest

from path_setup import ensure_src_path

ensure_src_path()

from backends.openai import OpenAIBackendConfig, OpenAIRewriteModel
from pipelines import ChunkWiseRephrasePipeline, PipelineConfig
from tokenization import WhitespaceTokenizer


def _require_live_api_or_skip(test_case: unittest.TestCase) -> None:
    if os.getenv("RUN_LIVE_LLM_TESTS", "").strip() != "1":
        test_case.skipTest("set RUN_LIVE_LLM_TESTS=1 to run live API integration tests")
    if not os.getenv("LLM_API_KEY", "").strip():
        test_case.skipTest("set LLM_API_KEY to run live API integration tests")


class OpenAIBackendLiveTests(unittest.TestCase):
    def test_pipeline_runs_with_real_llm_api(self) -> None:
        _require_live_api_or_skip(self)

        sample_path = (
            Path(__file__).resolve().parent / "data" / "live_rephrase_input.txt"
        )
        source_text = sample_path.read_text(encoding="utf-8")

        model = OpenAIRewriteModel(
            config=OpenAIBackendConfig(
                temperature=0.2,
                top_p=0.9,
                max_new_tokens=240,
            )
        )
        pipeline = ChunkWiseRephrasePipeline(
            model=model,
            tokenizer=WhitespaceTokenizer(),
            config=PipelineConfig(
                chunk_tokens=80,
                overlap_tokens=16,
                prefix_window_tokens=160,
                fidelity_threshold=0.0,
                max_retries=1,
                global_anchor_mode="head",
            ),
        )

        rewritten = pipeline.run(
            source_text,
            style_instruction="Rewrite for clarity while preserving facts, numbers, and named entities.",
        )

        self.assertIsInstance(rewritten, str)
        self.assertGreater(len(rewritten.strip()), 40)


if __name__ == "__main__":
    unittest.main()

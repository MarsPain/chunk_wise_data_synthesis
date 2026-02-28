import unittest

from path_setup import ensure_src_path

ensure_src_path()


class PipelinesApiTests(unittest.TestCase):
    def test_pipelines_package_exports_rephrase_and_generation(self) -> None:
        from pipelines import (
            ChunkWiseGenerationPipeline,
            ChunkWiseRephrasePipeline,
            PipelineConfig,
        )

        self.assertTrue(callable(ChunkWiseRephrasePipeline))
        self.assertTrue(callable(ChunkWiseGenerationPipeline))
        self.assertTrue(callable(PipelineConfig))

    def test_pipelines_package_exports_stitching_helpers(self) -> None:
        from pipelines import _longest_overlap, stitch_rewritten_chunks
        from tokenization import WhitespaceTokenizer

        overlap = _longest_overlap(
            left_tokens=["alpha", "beta", "gamma"],
            right_tokens=["gamma", "delta"],
            max_overlap_tokens=4,
        )
        self.assertEqual(overlap, 1)

        merged = stitch_rewritten_chunks(
            chunks=["alpha beta gamma", "gamma delta epsilon"],
            tokenizer=WhitespaceTokenizer(),
            max_overlap_tokens=4,
        )
        self.assertEqual(merged, "alpha beta gamma delta epsilon")


if __name__ == "__main__":
    unittest.main()

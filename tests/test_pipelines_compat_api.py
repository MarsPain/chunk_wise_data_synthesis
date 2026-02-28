import unittest

from path_setup import ensure_src_path

ensure_src_path()


class PipelinesCompatApiTests(unittest.TestCase):
    def test_legacy_pipeline_module_reexports_pipelines_api(self) -> None:
        import pipeline as legacy_pipeline
        from pipelines import (
            ChunkWiseRephrasePipeline as NewChunkWiseRephrasePipeline,
            PipelineConfig as NewPipelineConfig,
            _longest_overlap as new_longest_overlap,
            stitch_rewritten_chunks as new_stitch_rewritten_chunks,
        )

        self.assertIs(legacy_pipeline.PipelineConfig, NewPipelineConfig)
        self.assertIs(legacy_pipeline.ChunkWiseRephrasePipeline, NewChunkWiseRephrasePipeline)
        self.assertIs(legacy_pipeline._longest_overlap, new_longest_overlap)
        self.assertIs(legacy_pipeline.stitch_rewritten_chunks, new_stitch_rewritten_chunks)

    def test_legacy_generation_pipeline_module_reexports_pipelines_api(self) -> None:
        import generation_pipeline as legacy_generation_pipeline
        from pipelines import ChunkWiseGenerationPipeline as NewChunkWiseGenerationPipeline

        self.assertIs(legacy_generation_pipeline.ChunkWiseGenerationPipeline, NewChunkWiseGenerationPipeline)


if __name__ == "__main__":
    unittest.main()

import unittest

from path_setup import ensure_src_path

ensure_src_path()

from pipelines import (
    ChunkWiseGenerationPipeline,
    ChunkWiseRephrasePipeline,
    PipelineConfig,
    _longest_overlap,
    stitch_rewritten_chunks,
)
from pipelines.base import (
    _longest_overlap as longest_overlap_impl,
    stitch_rewritten_chunks as stitch_rewritten_chunks_impl,
)
from pipelines.generation import ChunkWiseGenerationPipeline as generation_pipeline_impl
from pipelines.rephrase import (
    ChunkWiseRephrasePipeline as rephrase_pipeline_impl,
    PipelineConfig as pipeline_config_impl,
)


class PipelinesApiTests(unittest.TestCase):
    def test_pipelines_package_exports_rephrase_api(self) -> None:
        self.assertIs(PipelineConfig, pipeline_config_impl)
        self.assertIs(ChunkWiseRephrasePipeline, rephrase_pipeline_impl)
        self.assertIs(_longest_overlap, longest_overlap_impl)
        self.assertIs(stitch_rewritten_chunks, stitch_rewritten_chunks_impl)

    def test_pipelines_package_exports_generation_api(self) -> None:
        self.assertIs(ChunkWiseGenerationPipeline, generation_pipeline_impl)


if __name__ == "__main__":
    unittest.main()

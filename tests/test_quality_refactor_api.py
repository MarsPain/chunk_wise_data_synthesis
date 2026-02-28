import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class QualityLayerApiTests(unittest.TestCase):
    def test_quality_base_helpers_available(self) -> None:
        from quality.base import _token_jaccard, _tokenize, _words_in_order

        self.assertEqual(_tokenize("State table state-table"), {"state", "table", "state-table"})
        self.assertAlmostEqual(_token_jaccard("a b", "a c"), 1 / 3)
        self.assertTrue(_words_in_order(["state", "table"], "A state, table keeps context."))

    def test_quality_package_exports_grouped_api(self) -> None:
        from quality import (
            CompositeFidelityVerifier,
            EntityPresenceChecker,
            FidelityVerifier,
            NoOpVerifier,
            NumericFactChecker,
            OutlineCoverageChecker,
            RepetitionAndDriftChecker,
            StrictConsistencyEditGuard,
            TerminologyConsistencyChecker,
            TokenJaccardVerifier,
        )

        self.assertTrue(callable(CompositeFidelityVerifier))
        self.assertTrue(callable(EntityPresenceChecker))
        self.assertTrue(callable(FidelityVerifier))
        self.assertTrue(callable(NoOpVerifier))
        self.assertTrue(callable(NumericFactChecker))
        self.assertTrue(callable(OutlineCoverageChecker))
        self.assertTrue(callable(RepetitionAndDriftChecker))
        self.assertTrue(callable(StrictConsistencyEditGuard))
        self.assertTrue(callable(TerminologyConsistencyChecker))
        self.assertTrue(callable(TokenJaccardVerifier))

    def test_generation_pipeline_no_longer_defines_duplicate_text_helpers(self) -> None:
        from pipelines import ChunkWiseGenerationPipeline

        self.assertFalse(hasattr(ChunkWiseGenerationPipeline, "_token_jaccard"))
        self.assertFalse(hasattr(ChunkWiseGenerationPipeline, "_words_in_order"))
        self.assertFalse(hasattr(ChunkWiseGenerationPipeline, "_get_missing_entities"))


if __name__ == "__main__":
    unittest.main()

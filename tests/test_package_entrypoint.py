import importlib
import sys
import unittest
from pathlib import Path


class PackageEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def test_src_package_reexports_public_api(self) -> None:
        import src

        self.assertTrue(callable(src.ChunkWiseRephrasePipeline))
        self.assertTrue(callable(src.PipelineConfig))
        self.assertTrue(callable(src.ChunkWiseGenerationPipeline))

        self.assertTrue(callable(src.RewriteRequest))
        self.assertTrue(callable(src.render_rewrite_prompt))
        self.assertTrue(callable(src.render_plan_prompt))

        self.assertTrue(callable(src.FidelityVerifier))
        self.assertTrue(callable(src.NumericFactChecker))
        self.assertTrue(callable(src.WhitespaceTokenizer))

        self.assertTrue(callable(src.OpenAIBackendConfig))
        self.assertTrue(callable(src.OpenAILLMModel))
        self.assertTrue(callable(src.OpenAIRewriteModel))

    def test_legacy_wrapper_modules_removed(self) -> None:
        legacy_modules = [
            "pipeline",
            "prompting",
            "fidelity",
            "generation_pipeline",
            "generation_prompting",
            "generation_quality",
            "tokenizer",
            "openai_backend",
        ]
        for module_name in legacy_modules:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()

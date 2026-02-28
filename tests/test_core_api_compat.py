import unittest

from path_setup import ensure_src_path

ensure_src_path()

from fidelity import FidelityVerifier as LegacyFidelityVerifier
from model import LLMModel as LegacyLLMModel
from model import LLMRequest as LegacyLLMRequest
from model import LLMTask as LegacyLLMTask
from model import RewriteModel as LegacyRewriteModel
from openai_backend import OpenAIBackendConfig as LegacyOpenAIBackendConfig
from pipeline import PipelineConfig as LegacyPipelineConfig
from prompting import PromptLanguage as LegacyPromptLanguage
from prompting import RewriteRequest as LegacyRewriteRequest
from tokenizer import Tokenizer as LegacyTokenizer
from generation_types import GenerationConfig as LegacyGenerationConfig
from generation_types import GenerationPlan as LegacyGenerationPlan
from generation_types import GenerationResult as LegacyGenerationResult
from generation_types import GenerationState as LegacyGenerationState
from generation_types import QualityReport as LegacyQualityReport
from generation_types import SectionSpec as LegacySectionSpec

from core import config as core_config
from core import protocols as core_protocols
from core import types as core_types


class CoreApiCompatTests(unittest.TestCase):
    def test_protocols_reexport_legacy_contracts(self) -> None:
        self.assertIs(core_protocols.Tokenizer, LegacyTokenizer)
        self.assertIs(core_protocols.LLMModel, LegacyLLMModel)
        self.assertIs(core_protocols.RewriteModel, LegacyRewriteModel)
        self.assertIs(core_protocols.FidelityVerifier, LegacyFidelityVerifier)

    def test_types_reexport_legacy_types(self) -> None:
        self.assertIs(core_types.LLMTask, LegacyLLMTask)
        self.assertIs(core_types.LLMRequest, LegacyLLMRequest)
        self.assertIs(core_types.RewriteRequest, LegacyRewriteRequest)
        self.assertIs(core_types.PromptLanguage, LegacyPromptLanguage)
        self.assertIs(core_types.GenerationPlan, LegacyGenerationPlan)
        self.assertIs(core_types.SectionSpec, LegacySectionSpec)
        self.assertIs(core_types.GenerationState, LegacyGenerationState)
        self.assertIs(core_types.GenerationResult, LegacyGenerationResult)
        self.assertIs(core_types.QualityReport, LegacyQualityReport)

    def test_config_reexport_legacy_configs(self) -> None:
        self.assertIs(core_config.PipelineConfig, LegacyPipelineConfig)
        self.assertIs(core_config.GenerationConfig, LegacyGenerationConfig)
        self.assertIs(core_config.OpenAIBackendConfig, LegacyOpenAIBackendConfig)

    def test_core_init_exposes_expected_modules(self) -> None:
        from core import __all__ as core_all

        self.assertEqual(core_all, ["protocols", "types", "config"])


if __name__ == "__main__":
    unittest.main()

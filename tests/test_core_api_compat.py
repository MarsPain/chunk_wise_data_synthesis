import unittest

from path_setup import ensure_src_path

ensure_src_path()

from backends.openai import OpenAIBackendConfig as OpenAIBackendConfigImpl
from generation_types import GenerationConfig, GenerationPlan, GenerationResult, GenerationState, QualityReport, SectionSpec
from model import LLMModel, LLMRequest, LLMTask, RewriteModel
from pipelines.rephrase import PipelineConfig as PipelineConfigImpl
from prompts.base import PromptLanguage
from prompts.rephrase import RewriteRequest
from quality.fidelity import FidelityVerifier
from tokenization import Tokenizer

from core import config as core_config
from core import protocols as core_protocols
from core import types as core_types


class CoreApiTests(unittest.TestCase):
    def test_protocols_reexport_contracts(self) -> None:
        self.assertIs(core_protocols.Tokenizer, Tokenizer)
        self.assertIs(core_protocols.LLMModel, LLMModel)
        self.assertIs(core_protocols.RewriteModel, RewriteModel)
        self.assertIs(core_protocols.FidelityVerifier, FidelityVerifier)

    def test_types_reexport_domain_types(self) -> None:
        self.assertIs(core_types.LLMTask, LLMTask)
        self.assertIs(core_types.LLMRequest, LLMRequest)
        self.assertIs(core_types.RewriteRequest, RewriteRequest)
        self.assertIs(core_types.PromptLanguage, PromptLanguage)
        self.assertIs(core_types.GenerationPlan, GenerationPlan)
        self.assertIs(core_types.SectionSpec, SectionSpec)
        self.assertIs(core_types.GenerationState, GenerationState)
        self.assertIs(core_types.GenerationResult, GenerationResult)
        self.assertIs(core_types.QualityReport, QualityReport)

    def test_config_reexport_domain_configs(self) -> None:
        self.assertIs(core_config.PipelineConfig, PipelineConfigImpl)
        self.assertIs(core_config.GenerationConfig, GenerationConfig)
        self.assertIs(core_config.OpenAIBackendConfig, OpenAIBackendConfigImpl)

    def test_core_init_exposes_expected_modules(self) -> None:
        from core import __all__ as core_all

        self.assertEqual(core_all, ["protocols", "types", "config"])


if __name__ == "__main__":
    unittest.main()

import unittest

from path_setup import ensure_src_path

ensure_src_path()

from backends import OpenAIBackendConfig, OpenAILLMModel, OpenAIRewriteModel
from backends.openai import OpenAIBackendConfig as NewOpenAIBackendConfig
from backends.openai import OpenAILLMModel as NewOpenAILLMModel
from backends.openai import OpenAIRewriteModel as NewOpenAIRewriteModel
from openai_backend import OpenAIBackendConfig as LegacyOpenAIBackendConfig
from openai_backend import OpenAILLMModel as LegacyOpenAILLMModel
from openai_backend import OpenAIRewriteModel as LegacyOpenAIRewriteModel


class BackendsApiCompatTests(unittest.TestCase):
    def test_backends_package_exports_openai_symbols(self) -> None:
        self.assertIs(OpenAIBackendConfig, NewOpenAIBackendConfig)
        self.assertIs(OpenAILLMModel, NewOpenAILLMModel)
        self.assertIs(OpenAIRewriteModel, NewOpenAIRewriteModel)

    def test_legacy_openai_backend_reexports_new_symbols(self) -> None:
        self.assertIs(LegacyOpenAIBackendConfig, NewOpenAIBackendConfig)
        self.assertIs(LegacyOpenAILLMModel, NewOpenAILLMModel)
        self.assertIs(LegacyOpenAIRewriteModel, NewOpenAIRewriteModel)


if __name__ == "__main__":
    unittest.main()

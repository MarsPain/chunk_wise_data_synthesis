import unittest

from path_setup import ensure_src_path

ensure_src_path()

from backends import OpenAIBackendConfig, OpenAILLMModel, OpenAIRewriteModel
from backends.openai import OpenAIBackendConfig as OpenAIBackendConfigImpl
from backends.openai import OpenAILLMModel as OpenAILLMModelImpl
from backends.openai import OpenAIRewriteModel as OpenAIRewriteModelImpl


class BackendsApiTests(unittest.TestCase):
    def test_backends_package_exports_openai_symbols(self) -> None:
        self.assertIs(OpenAIBackendConfig, OpenAIBackendConfigImpl)
        self.assertIs(OpenAILLMModel, OpenAILLMModelImpl)
        self.assertIs(OpenAIRewriteModel, OpenAIRewriteModelImpl)


if __name__ == "__main__":
    unittest.main()

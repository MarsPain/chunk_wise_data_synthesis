import unittest
from types import SimpleNamespace

from path_setup import ensure_src_path

ensure_src_path()

from openai_backend import OpenAIBackendConfig, OpenAILLMModel, OpenAIRewriteModel
from model import LLMRequest
from prompting import RewriteRequest


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" generated output "))]
        )


class ModelCompatTests(unittest.TestCase):
    def test_openai_llm_model_forwards_prompt(self) -> None:
        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test", model="m", temperature=0.3, top_p=0.8),
            client=fake_client,
        )

        output = model.generate(LLMRequest(task="section_generation", prompt="section prompt"))

        self.assertEqual(output, "generated output")
        self.assertEqual(fake_completions.last_kwargs["messages"][0]["content"], "section prompt")

    def test_openai_rewrite_model_stays_backward_compatible(self) -> None:
        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        model = OpenAIRewriteModel(
            config=OpenAIBackendConfig(api_key="test", model="rewrite-model"),
            client=fake_client,
        )
        request = RewriteRequest(
            style_instruction="encyclopedic",
            global_anchor="anchor",
            generated_prefix="prefix",
            current_chunk="chunk",
            retry_index=0,
            strict_fidelity=False,
        )

        output = model.rewrite(request)

        self.assertEqual(output, "generated output")
        self.assertEqual(fake_completions.last_kwargs["model"], "rewrite-model")
        prompt = fake_completions.last_kwargs["messages"][0]["content"]
        self.assertIn("Current chunk:", prompt)
        self.assertIn("chunk", prompt)


if __name__ == "__main__":
    unittest.main()

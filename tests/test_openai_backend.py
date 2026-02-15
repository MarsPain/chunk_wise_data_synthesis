import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from path_setup import ensure_src_path

ensure_src_path()

from openai_backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAIBackendConfig,
    OpenAIRewriteModel,
)
from prompting import RewriteRequest

# 配置日志
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        logger.debug(f"[FakeCompletions] 收到调用参数: {kwargs}")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" rewritten chunk "))]
        )
        logger.debug(f"[FakeCompletions] 返回响应: {response}")
        return response


class _ErrorCompletions:
    def create(self, **kwargs):
        raise RuntimeError("step-3.5-flash:free is not a valid model ID")


class OpenAIBackendTests(unittest.TestCase):
    def test_defaults_use_openrouter_and_env_key(self) -> None:
        logger.info("=== 测试: 默认使用 OpenRouter 和环境变量 API Key ===")
        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        captured: dict[str, str] = {}

        def client_factory(api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            logger.info(f"[ClientFactory] 被调用 - api_key: {api_key[:8]}..., base_url: {base_url}")
            return fake_client

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            model = OpenAIRewriteModel(client_factory=client_factory)
            request = RewriteRequest(
                style_instruction="encyclopedic",
                global_anchor="anchor",
                generated_prefix="prefix",
                current_chunk="source text",
                retry_index=0,
                strict_fidelity=False,
            )
            logger.info(f"[Test] 发送 rewrite 请求: style={request.style_instruction}, chunk={request.current_chunk}")
            result = model.rewrite(request)
            logger.info(f"[Test] 收到 rewrite 结果: '{result}'")

        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], DEFAULT_BASE_URL)
        self.assertEqual(result, "rewritten chunk")
        self.assertIsNotNone(fake_completions.last_kwargs)
        self.assertEqual(fake_completions.last_kwargs["model"], DEFAULT_MODEL)
        logger.info("=== 测试通过 ===\n")

    def test_rewrite_forwards_generation_parameters(self) -> None:
        logger.info("=== 测试: 生成参数正确传递 ===")
        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        config = OpenAIBackendConfig(
            api_key="direct-key",
            base_url="https://example.com/v1",
            model="custom-model",
            temperature=0.25,
            top_p=0.9,
            max_new_tokens=300,
        )
        logger.info(f"[Test] 配置: model={config.model}, temperature={config.temperature}, top_p={config.top_p}")
        model = OpenAIRewriteModel(config=config, client=fake_client)
        request = RewriteRequest(
            style_instruction="qa",
            global_anchor="global facts",
            generated_prefix="generated before",
            current_chunk="current chunk",
            retry_index=1,
            strict_fidelity=True,
        )

        _ = model.rewrite(request)

        self.assertIsNotNone(fake_completions.last_kwargs)
        self.assertEqual(fake_completions.last_kwargs["model"], "custom-model")
        self.assertEqual(fake_completions.last_kwargs["temperature"], 0.25)
        self.assertEqual(fake_completions.last_kwargs["top_p"], 0.9)
        self.assertEqual(fake_completions.last_kwargs["max_tokens"], 300)
        self.assertIn("Current chunk:", fake_completions.last_kwargs["messages"][0]["content"])
        self.assertIn("generated before", fake_completions.last_kwargs["messages"][0]["content"])
        logger.info(f"[Test] 验证的 prompt 内容包含: {fake_completions.last_kwargs['messages'][0]['content'][:100]}...")
        logger.info("=== 测试通过 ===\n")

    def test_missing_env_key_raises_clear_error(self) -> None:
        logger.info("=== 测试: 缺失环境变量时报错 ===")

        def client_factory(api_key: str, base_url: str):
            return SimpleNamespace()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY") as cm:
                OpenAIRewriteModel(client_factory=client_factory)
            logger.info(f"[Test] 捕获到预期异常: {cm.exception}")
        logger.info("=== 测试通过 ===\n")

    def test_env_model_overrides_default_model(self) -> None:
        logger.info("=== 测试: LLM_MODEL 可覆盖默认模型 ===")
        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))

        def client_factory(api_key: str, base_url: str):
            return fake_client

        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_MODEL": "openai/gpt-4o-mini"},
            clear=True,
        ):
            model = OpenAIRewriteModel(client_factory=client_factory)
            request = RewriteRequest(
                style_instruction="encyclopedic",
                global_anchor="anchor",
                generated_prefix="prefix",
                current_chunk="source text",
                retry_index=0,
                strict_fidelity=False,
            )
            _ = model.rewrite(request)

        self.assertEqual(fake_completions.last_kwargs["model"], "openai/gpt-4o-mini")
        logger.info("=== 测试通过 ===\n")

    def test_invalid_model_error_contains_override_hint(self) -> None:
        logger.info("=== 测试: 无效模型报错包含 LLM_MODEL 提示 ===")
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_ErrorCompletions()))
        model = OpenAIRewriteModel(
            config=OpenAIBackendConfig(api_key="x"),
            client=fake_client,
        )
        request = RewriteRequest(
            style_instruction="encyclopedic",
            global_anchor="anchor",
            generated_prefix="prefix",
            current_chunk="source text",
            retry_index=0,
            strict_fidelity=False,
        )

        with self.assertRaisesRegex(ValueError, "LLM_MODEL"):
            _ = model.rewrite(request)
        logger.info("=== 测试通过 ===\n")


if __name__ == "__main__":
    unittest.main()

import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from path_setup import ensure_src_path

ensure_src_path()

from backends.openai import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAIBackendConfig,
    OpenAIRewriteModel,
    OpenAILLMModel,
)
from model import LLMRequest
from prompts import RewriteRequest

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


class _FakeCompletionsWithErrors:
    """Mock completions that raises specific errors on each call."""
    def __init__(self, errors: list[Exception]) -> None:
        self._errors = errors
        self._call_count = 0
    
    def create(self, **kwargs):
        if self._call_count < len(self._errors):
            error = self._errors[self._call_count]
            self._call_count += 1
            raise error
        # Return success on the call after errors are exhausted
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" success "))]
        )


class _FakeRateLimitError(Exception):
    """Fake RateLimitError for testing."""
    pass


class _FakeAPIStatusError(Exception):
    """Fake APIStatusError with status_code for testing.
    
    Mimics openai.APIStatusError which has status_code attribute.
    """
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class _FakeTimeoutError(Exception):
    """Fake APITimeoutError for testing."""
    pass


def _make_fake_status_error(message: str, status_code: int) -> Exception:
    """Create a fake API status error with status_code attribute."""
    return _FakeAPIStatusError(message, status_code)


class OpenAIBackendRetryTests(unittest.TestCase):
    """Test status code-based retry logic."""

    def test_429_rate_limit_retries_with_exponential_backoff(self) -> None:
        """429 errors should be retried with exponential backoff."""
        logger.info("=== 测试: 429 错误指数退避重试 ===")
        
        # Mock errors with 429 status code
        fake_errors = [
            _make_fake_status_error("Rate limit exceeded", 429),
            _make_fake_status_error("Rate limit exceeded", 429),
        ]
        fake_completions = _FakeCompletionsWithErrors(fake_errors)
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        # Test with short delays for fast test
        result = model._generate_with_retry(
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            LLMRequest(task="test", prompt="hello"),
            max_retries=5,
            base_delay=0.01,  # Very short for fast test
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(fake_completions._call_count, 2)  # 2 retries before success
        logger.info("=== 测试通过: 429 正确重试 ===\n")

    def test_5xx_server_error_retries_then_succeeds(self) -> None:
        """5xx errors should be retried and eventually succeed."""
        logger.info("=== 测试: 5xx 错误重试后成功 ===")
        
        fake_errors = [
            _make_fake_status_error("Service unavailable", 503),
        ]
        fake_completions = _FakeCompletionsWithErrors(fake_errors)
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        result = model._generate_with_retry(
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            LLMRequest(task="test", prompt="hello"),
            max_retries=5,
            base_delay=0.01,
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(fake_completions._call_count, 1)  # 1 retry before success
        logger.info("=== 测试通过: 5xx 重试后成功 ===\n")

    def test_401_unauthorized_fails_fast_with_readable_message(self) -> None:
        """401 errors should fail immediately with a helpful message."""
        logger.info("=== 测试: 401 错误快速失败并返回可读信息 ===")
        
        fake_completions = _FakeCompletionsWithErrors([
            _make_fake_status_error("Invalid API key", 401),
        ])
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        with self.assertRaises(ValueError) as cm:
            model._generate_with_retry(
                {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                LLMRequest(task="test", prompt="hello"),
            )
        
        error_msg = str(cm.exception)
        self.assertIn("401", error_msg)
        self.assertIn("API Key", error_msg)
        # Should only be called once (no retries for 4xx)
        self.assertEqual(fake_completions._call_count, 1)
        logger.info(f"捕获到预期错误: {error_msg[:100]}...")
        logger.info("=== 测试通过: 401 快速失败 ===\n")

    def test_403_forbidden_fails_fast(self) -> None:
        """403 errors should fail immediately with a helpful message."""
        logger.info("=== 测试: 403 错误快速失败 ===")
        
        fake_completions = _FakeCompletionsWithErrors([
            _make_fake_status_error("Access denied", 403),
        ])
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        with self.assertRaises(ValueError) as cm:
            model._generate_with_retry(
                {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                LLMRequest(task="test", prompt="hello"),
            )
        
        error_msg = str(cm.exception)
        self.assertIn("403", error_msg)
        self.assertEqual(fake_completions._call_count, 1)
        logger.info(f"捕获到预期错误: {error_msg[:100]}...")
        logger.info("=== 测试通过: 403 快速失败 ===\n")

    def test_404_not_found_fails_fast(self) -> None:
        """404 errors should fail immediately with helpful message."""
        logger.info("=== 测试: 404 错误快速失败 ===")
        
        fake_completions = _FakeCompletionsWithErrors([
            _make_fake_status_error("Model not found", 404),
        ])
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        with self.assertRaises(ValueError) as cm:
            model._generate_with_retry(
                {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                LLMRequest(task="test", prompt="hello"),
            )
        
        error_msg = str(cm.exception)
        self.assertIn("404", error_msg)
        self.assertEqual(fake_completions._call_count, 1)
        logger.info(f"捕获到预期错误: {error_msg[:100]}...")
        logger.info("=== 测试通过: 404 快速失败 ===\n")

    def test_5xx_exhausts_retries_then_fails(self) -> None:
        """5xx errors that persist should eventually fail."""
        logger.info("=== 测试: 持续的 5xx 错误最终失败 ===")
        
        # All calls return 503
        fake_completions = MagicMock()
        fake_completions.create.side_effect = _make_fake_status_error(
            "Service unavailable", 503
        )
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        
        model = OpenAILLMModel(
            config=OpenAIBackendConfig(api_key="test-key"),
            client=fake_client,
        )
        
        with self.assertRaises(ValueError) as cm:
            model._generate_with_retry(
                {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                LLMRequest(task="test", prompt="hello"),
                max_retries=2,
                base_delay=0.01,  # Very short delay for fast test
            )
        
        error_msg = str(cm.exception)
        self.assertIn("503", error_msg)
        # Should have been called max_retries times
        self.assertEqual(fake_completions.create.call_count, 2)
        logger.info(f"捕获到预期错误: {error_msg[:100]}...")
        logger.info("=== 测试通过: 5xx 耗尽重试后失败 ===\n")


if __name__ == "__main__":
    unittest.main()

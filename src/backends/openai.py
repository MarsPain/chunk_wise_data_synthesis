from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from model import LLMModel, LLMRequest, RewriteModel
from prompts.rephrase import RewriteRequest, render_rewrite_prompt

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "stepfun/step-3.5-flash:free"
# DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_BASE_URL_ENV_VAR = "LLM_BASE_URL"
DEFAULT_MODEL_ENV_VAR = "LLM_MODEL"


@dataclass(frozen=True)
class OpenAIBackendConfig:
    api_key: str | None = None
    api_key_env_var: str = "LLM_API_KEY"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int | None = None
    # Reasoning mode: None for default, False to disable, True to enable
    # Some models (like stepfun/step-3.5-flash) output thinking content which can break JSON parsing
    reasoning: bool | None = None


def _default_client_factory(api_key: str, base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_message_content(message: Any) -> str:
    """Extract text content from message, handling content/reasoning fields."""
    # Try standard content field first
    content = getattr(message, "content", None)
    if content:
        text = _extract_text_content(content)
        if text:
            return text
    
    # Fallback to reasoning field (some models like stepfun return reasoning instead)
    reasoning = getattr(message, "reasoning", None)
    if reasoning and isinstance(reasoning, str):
        return reasoning
    
    return ""


class OpenAILLMModel(LLMModel):
    def __init__(
        self,
        config: OpenAIBackendConfig | None = None,
        client: Any | None = None,
        client_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self._config = config or OpenAIBackendConfig()
        self._base_url = self._resolve_base_url(self._config)
        self._model = self._resolve_model(self._config)
        if client is not None:
            self._client = client
            return

        api_key = self._resolve_api_key(self._config)
        factory = client_factory or _default_client_factory
        self._client = factory(api_key, self._base_url)

    def generate(self, request: LLMRequest) -> str:
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
        }
        if self._config.max_new_tokens is not None:
            create_kwargs["max_tokens"] = self._config.max_new_tokens
        
        return self._generate_with_retry(create_kwargs, request)

    def _generate_with_retry(
        self,
        create_kwargs: dict[str, Any],
        request: LLMRequest,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> str:
        """Generate with exponential backoff retry for rate limiting and 5xx errors.
        
        Retry policy:
        - 429 (RateLimitError): exponential backoff retry
        - 5xx (server errors): exponential backoff retry
        - Timeout: exponential backoff retry
        - 4xx (client errors, except 429): fast fail with readable message
        """
        import logging

        logger = logging.getLogger(__name__)

        # Try to import specific exception types from openai module
        try:
            import openai
            RateLimitError = openai.RateLimitError
            APIStatusError = openai.APIStatusError
            APITimeoutError = openai.APITimeoutError
        except Exception:
            RateLimitError = None
            APIStatusError = None
            APITimeoutError = None

        prompt = create_kwargs.get("messages", [{}])[0].get("content", "")
        prompt_len = len(prompt)
        logger.info(f"[LLM] [{request.task}] Request: prompt_len={prompt_len} chars")

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self._client.chat.completions.create(**create_kwargs)
                elapsed = time.time() - start_time

                message = response.choices[0].message
                content = _extract_message_content(message)
                result = content.strip()

                logger.info(f"[LLM] [{request.task}] Response: len={len(result)} chars, time={elapsed:.2f}s")
                logger.debug(f"[LLM] [{request.task}] Raw output:\n{result[:500]}{'...' if len(result) > 500 else ''}")

                return result

            except Exception as exc:
                # Get status code if available (from APIStatusError or mock objects)
                status = getattr(exc, 'status_code', None)
                exc_str = str(exc)
                
                # Handle rate limit errors (429) - check by type first, then status code, then string
                is_rate_limit = (
                    (RateLimitError is not None and isinstance(exc, RateLimitError)) or
                    (status == 429) or
                    ("429" in exc_str or "rate limit" in exc_str.lower())
                )
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"429 Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    raise

                # Handle timeout errors
                is_timeout = (
                    (APITimeoutError is not None and isinstance(exc, APITimeoutError)) or
                    ("timeout" in exc_str.lower())
                )
                if is_timeout:
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    raise ValueError(f"[{request.task}] Request timeout after {max_retries} attempts") from exc

                # Handle API status errors (includes 4xx and 5xx) - check by type or status code
                is_api_status_error = (
                    (APIStatusError is not None and isinstance(exc, APIStatusError)) or
                    (status is not None)
                )
                if is_api_status_error and status is not None:
                    # 5xx server errors: retry with exponential backoff
                    if status >= 500:
                        if attempt < max_retries - 1:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            logger.warning(f"{status} Server error (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        raise ValueError(self._format_client_error(status, exc, request.task)) from exc
                    
                    # 4xx client errors (except 429 which is handled above): fast fail
                    if 400 <= status < 500:
                        raise ValueError(self._format_client_error(status, exc, request.task)) from exc
                
                # Fallback: Check for 5xx in error message
                if any(f"{code}" in exc_str for code in [500, 502, 503, 504]):
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Server error (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                
                # Check for invalid model ID (special case handling)
                if "valid model ID" in exc_str:
                    raise ValueError(
                        f"Invalid model id for current provider. "
                        f"Please set {DEFAULT_MODEL_ENV_VAR} to a valid model id (current: {self._model})."
                    ) from exc
                
                # All other errors: raise immediately
                raise

        raise RuntimeError(f"[{request.task}] Max retries exceeded")

    def _format_client_error(self, status: int, exc: Exception, task: str) -> str:
        """Format client errors into human-readable messages."""
        error_map = {
            400: "Bad request (400): 请求参数错误",
            401: "Unauthorized (401): API Key 无效或已过期，请检查 LLM_API_KEY",
            403: "Forbidden (403): 无权访问该模型，请检查模型权限",
            404: "Not found (404): 模型不存在，请检查 LLM_MODEL 设置",
            422: "Unprocessable entity (422): 请求格式错误",
            429: "Rate limited (429): 请求过于频繁，请稍后再试",
            500: "Internal server error (500): 服务器内部错误",
            502: "Bad gateway (502): 网关错误",
            503: "Service unavailable (503): 服务暂时不可用",
            504: "Gateway timeout (504): 网关超时",
        }
        
        desc = error_map.get(status, f"HTTP {status} error")
        detail = str(exc)
        
        # Special case for invalid model ID in error message
        if "valid model ID" in detail:
            return (
                f"[{task}] Invalid model id: 当前模型 '{self._model}' 不被提供商支持，"
                f"请设置有效的 {DEFAULT_MODEL_ENV_VAR}"
            )
        
        return f"[{task}] {desc}. Detail: {detail[:200]}"

    @staticmethod
    def _resolve_api_key(config: OpenAIBackendConfig) -> str:
        if config.api_key:
            return config.api_key
        env_value = os.getenv(config.api_key_env_var, "").strip()
        if env_value:
            return env_value
        raise ValueError(
            f"Missing API key. Set {config.api_key_env_var} or pass api_key in OpenAIBackendConfig."
        )

    @staticmethod
    def _resolve_base_url(config: OpenAIBackendConfig) -> str:
        env_base_url = os.getenv(DEFAULT_BASE_URL_ENV_VAR, "").strip()
        if env_base_url and config.base_url == DEFAULT_BASE_URL:
            return env_base_url
        return config.base_url

    @staticmethod
    def _resolve_model(config: OpenAIBackendConfig) -> str:
        env_model = os.getenv(DEFAULT_MODEL_ENV_VAR, "").strip()
        if env_model and config.model == DEFAULT_MODEL:
            return env_model
        return config.model


class OpenAIRewriteModel(RewriteModel):
    """Backward-compatible wrapper around OpenAILLMModel."""

    def __init__(
        self,
        config: OpenAIBackendConfig | None = None,
        client: Any | None = None,
        client_factory: Callable[[str, str], Any] | None = None,
        llm_model: OpenAILLMModel | None = None,
    ) -> None:
        if llm_model is not None:
            self._llm_model = llm_model
            return
        self._llm_model = OpenAILLMModel(
            config=config,
            client=client,
            client_factory=client_factory,
        )

    def rewrite(self, request: RewriteRequest) -> str:
        prompt = render_rewrite_prompt(request)
        return self._llm_model.generate(
            LLMRequest(task="rewrite_chunk", prompt=prompt)
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_BASE_URL_ENV_VAR",
    "DEFAULT_MODEL_ENV_VAR",
    "OpenAIBackendConfig",
    "OpenAILLMModel",
    "OpenAIRewriteModel",
]

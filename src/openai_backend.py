from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from model import LLMModel, LLMRequest, RewriteModel
from prompting import RewriteRequest, render_rewrite_prompt

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
        base_delay: float = 20.0,
        max_delay: float = 60.0,
    ) -> str:
        """Generate with exponential backoff retry for rate limiting."""
        import logging

        logger = logging.getLogger(__name__)

        # Try to get RateLimitError from openai module if available
        try:
            import openai
            rate_limit_error = openai.RateLimitError
        except Exception:
            rate_limit_error = None

        last_exception: Exception | None = None
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
                last_exception = exc
                # Check if it's a rate limit error (either by type or error message)
                is_rate_limit = (
                    (rate_limit_error is not None and isinstance(exc, rate_limit_error)) or
                    ("429" in str(exc) or "rate limit" in str(exc).lower())
                )
                if is_rate_limit and attempt < max_retries - 1:
                    # Exponential backoff: 5s, 10s, 20s, 40s, 60s...
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                message = str(exc)
                if "valid model ID" in message:
                    raise ValueError(
                        "Invalid model id for current provider. "
                        f"Please set {DEFAULT_MODEL_ENV_VAR} to a valid model id (current: {self._model})."
                    ) from exc
                raise

        raise last_exception or RuntimeError("Max retries exceeded for rate limiting")

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

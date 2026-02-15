from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from model import RewriteModel
from prompting import RewriteRequest, render_rewrite_prompt

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "stepfun/step-3.5-flash:free"
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


class OpenAIRewriteModel(RewriteModel):
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

    def rewrite(self, request: RewriteRequest) -> str:
        prompt = render_rewrite_prompt(request)
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
        }
        if self._config.max_new_tokens is not None:
            create_kwargs["max_tokens"] = self._config.max_new_tokens

        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            message = str(exc)
            if "valid model ID" in message:
                raise ValueError(
                    "Invalid model id for current provider. "
                    f"Please set {DEFAULT_MODEL_ENV_VAR} to a valid model id (current: {self._model})."
                ) from exc
            raise
        message = response.choices[0].message
        content = _extract_text_content(getattr(message, "content", ""))
        return content.strip()

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

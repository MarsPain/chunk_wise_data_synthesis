from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from prompts.rephrase import RewriteRequest


LLMTask = Literal[
    "rewrite_chunk",
    "plan_generation",
    "section_generation",
    "consistency_pass",
]


@dataclass(frozen=True)
class LLMRequest:
    task: LLMTask
    prompt: str


class LLMModel(Protocol):
    def generate(self, request: LLMRequest) -> str:
        ...


class RewriteModel(Protocol):
    def rewrite(self, request: RewriteRequest) -> str:
        ...


class RewriteModelFromLLM:
    def __init__(self, llm_model: LLMModel) -> None:
        self._llm_model = llm_model

    def rewrite(self, request: RewriteRequest) -> str:
        from prompts.rephrase import render_rewrite_prompt

        prompt = render_rewrite_prompt(request)
        return self._llm_model.generate(
            LLMRequest(task="rewrite_chunk", prompt=prompt)
        )

from __future__ import annotations

from typing import Protocol

from prompting import RewriteRequest


class RewriteModel(Protocol):
    def rewrite(self, request: RewriteRequest) -> str:
        ...

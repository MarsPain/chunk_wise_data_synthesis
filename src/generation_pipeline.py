from __future__ import annotations

"""Legacy generation pipeline module kept for backward compatibility.

Canonical implementation now lives in `pipelines.generation`.
"""

from pipelines.generation import ChunkWiseGenerationPipeline

__all__ = [
    "ChunkWiseGenerationPipeline",
]

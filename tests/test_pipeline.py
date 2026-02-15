import unittest
from dataclasses import dataclass
from typing import List

from path_setup import ensure_src_path

ensure_src_path()

from pipeline import ChunkWiseRephrasePipeline, PipelineConfig
from prompting import RewriteRequest
from tokenizer import WhitespaceTokenizer


@dataclass
class _RecordedCall:
    request: RewriteRequest


class ScriptedRewriteModel:
    def __init__(self, outputs: List[str]) -> None:
        self._outputs = outputs
        self.calls: List[_RecordedCall] = []
        self._cursor = 0

    def rewrite(self, request: RewriteRequest) -> str:
        self.calls.append(_RecordedCall(request=request))
        if self._cursor >= len(self._outputs):
            raise RuntimeError("scripted outputs exhausted")
        value = self._outputs[self._cursor]
        self._cursor += 1
        return value


class ContainsGoodVerifier:
    def score(self, source_text: str, rewritten_text: str) -> float:
        if "good" in rewritten_text:
            return 0.95
        return 0.10


class PipelineTests(unittest.TestCase):
    def test_autoregressive_prefix_passed_to_next_chunk(self) -> None:
        tokenizer = WhitespaceTokenizer()
        model = ScriptedRewriteModel(outputs=["R0 R1 R2 R3", "R4 R5 R6 R7"])
        config = PipelineConfig(
            chunk_tokens=4,
            overlap_tokens=0,
            prefix_window_tokens=16,
            fidelity_threshold=0.0,
            max_retries=1,
            global_anchor_mode="none",
        )
        pipeline = ChunkWiseRephrasePipeline(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        output = pipeline.run("t0 t1 t2 t3 t4 t5 t6 t7", style_instruction="encyclopedic")

        self.assertEqual(output, "R0 R1 R2 R3 R4 R5 R6 R7")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0].request.generated_prefix, "")
        self.assertEqual(model.calls[1].request.generated_prefix, "R0 R1 R2 R3")
        self.assertEqual(model.calls[1].request.style_instruction, "encyclopedic")

    def test_fidelity_failure_retries_until_threshold(self) -> None:
        tokenizer = WhitespaceTokenizer()
        model = ScriptedRewriteModel(outputs=["bad rewrite", "good rewrite"])
        config = PipelineConfig(
            chunk_tokens=8,
            overlap_tokens=0,
            prefix_window_tokens=8,
            fidelity_threshold=0.8,
            max_retries=2,
            global_anchor_mode="none",
        )
        pipeline = ChunkWiseRephrasePipeline(
            model=model,
            tokenizer=tokenizer,
            config=config,
            verifier=ContainsGoodVerifier(),
        )

        output = pipeline.run("s0 s1 s2 s3")

        self.assertEqual(output, "good rewrite")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0].request.retry_index, 0)
        self.assertEqual(model.calls[1].request.retry_index, 1)


if __name__ == "__main__":
    unittest.main()

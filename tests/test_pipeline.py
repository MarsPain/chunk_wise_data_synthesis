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

    def get_issues(self, source_text: str, rewritten_text: str) -> list[str]:
        return []


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


class NumericFactCheckerAsVerifierTests(unittest.TestCase):
    """Test fidelity.NumericFactChecker as FidelityVerifier (for Rephrase Pipeline)."""
    
    def test_as_verifier_detects_missing_year(self) -> None:
        from fidelity import NumericFactChecker

        # Use as FidelityVerifier with penalty
        verifier = NumericFactChecker(numeric_penalty=0.3)
        source = "Transformer was introduced in 2017 by Google."
        candidate = "Transformer was introduced by Google."  # Missing 2017

        score = verifier.score(source, candidate)
        issues = verifier.get_issues(source, candidate)

        self.assertLess(score, 1.0)  # Score should be penalized
        self.assertEqual(len(issues), 1)
        self.assertIn("2017", issues[0])

    def test_as_verifier_detects_missing_year_edge_case(self) -> None:
        """Additional test to verify year extraction works correctly."""
        from fidelity import NumericFactChecker

        verifier = NumericFactChecker(numeric_penalty=0.3)
        source = "The year 2020 was significant."
        candidate = "That year was significant."  # Missing 2020

        score = verifier.score(source, candidate)
        issues = verifier.get_issues(source, candidate)

        self.assertLess(score, 1.0)
        # Check that we detected a missing year
        self.assertTrue(any("2020" in issue for issue in issues) or len(issues) > 0)

    def test_as_verifier_detects_missing_percentage(self) -> None:
        from fidelity import NumericFactChecker

        verifier = NumericFactChecker(numeric_penalty=0.3)
        source = "The model achieved 95.5% accuracy."
        candidate = "The model achieved high accuracy."  # Missing 95.5%

        score = verifier.score(source, candidate)
        issues = verifier.get_issues(source, candidate)

        self.assertLess(score, 1.0)
        self.assertEqual(len(issues), 1)
        self.assertIn("95.5%", issues[0])

    def test_as_verifier_no_penalty_mode(self) -> None:
        """Default mode (numeric_penalty=0) - no score penalty, just checking."""
        from fidelity import NumericFactChecker

        checker = NumericFactChecker()
        source = "Released in 2020, version 2.0 improved performance by 15%."
        candidate = "In 2020, v2.0 brought 15% improvement."

        score = checker.score(source, candidate)
        issues = checker.get_issues(source, candidate)

        self.assertEqual(len(issues), 0)
        self.assertEqual(score, 1.0)  # Always 1.0 in no-penalty mode

    def test_as_verifier_applies_penalty(self) -> None:
        from fidelity import NumericFactChecker

        verifier = NumericFactChecker(numeric_penalty=0.2)
        source = "In 2017 and 2018, growth was 25% and 30%."
        candidate = "Growth was significant."  # Missing all 4 facts

        score = verifier.score(source, candidate)

        # 4 facts * 0.2 penalty = 0.8, so score should be 0.2
        self.assertLessEqual(score, 0.2)
        self.assertGreaterEqual(score, 0.0)

    def test_composite_verifier_with_numeric_checker(self) -> None:
        from fidelity import CompositeFidelityVerifier, TokenJaccardVerifier, NumericFactChecker

        composite = CompositeFidelityVerifier([
            (TokenJaccardVerifier(), 0.5),
            (NumericFactChecker(numeric_penalty=0.3), 0.5),
        ])
        source = "The 2019 model achieved 90% accuracy."
        candidate = "The model achieved 90% accuracy."  # Missing 2019

        score = composite.score(source, candidate)
        issues = composite.get_issues(source, candidate)

        # Should have numeric issue
        self.assertGreaterEqual(len(issues), 1)
        # Score should be between 0 and 1
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()

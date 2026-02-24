from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from generation_pipeline import ChunkWiseGenerationPipeline
from generation_types import GenerationConfig, GenerationPlan
from openai_backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAIBackendConfig,
    OpenAILLMModel,
)
from tokenizer import WhitespaceTokenizer


def _load_manual_plan(path: Path | None) -> GenerationPlan | None:
    if path is None:
        return None
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual plan file must contain a JSON object")
    return GenerationPlan.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chunk-wise autoregressive long-form generation with a real OpenAI-compatible API."
    )
    parser.add_argument("--topic", type=str, default="Chunk-wise autoregressive generation")
    parser.add_argument("--objective", type=str, default="Generate long-form training data")
    parser.add_argument("--audience", type=str, default="ML engineers")
    parser.add_argument("--tone", type=str, default="neutral technical")
    parser.add_argument("--target-tokens", type=int, default=1500)
    # parser.add_argument("--topic", type=str, default="分块自回归生成")
    # parser.add_argument("--objective", type=str, default="生成长篇训练数据")
    # parser.add_argument("--audience", type=str, default="机器学习工程师")
    # parser.add_argument("--tone", type=str, default="中性技术")
    parser.add_argument(
        "--manual-plan-path",
        type=Path,
        default=None,
        help="Optional JSON plan path. If provided, topic/objective fields are ignored.",
    )
    parser.add_argument(
        "--disable-consistency-pass",
        action="store_true",
        help="Disable the final consistency pass.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, prints generated text to stdout.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--prefix-window-tokens", type=int, default=1200)
    parser.add_argument(
        "--prompt-language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Prompt language for instructions sent to the model.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable reasoning mode for supported models. Default is disabled to avoid JSON parsing issues.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Preparing model and pipeline configuration")
    # For generation tasks, we need JSON output and disable reasoning to avoid parsing issues
    model = OpenAILLMModel(
        config=OpenAIBackendConfig(
            base_url=args.base_url or DEFAULT_BASE_URL,
            model=args.model or DEFAULT_MODEL,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            reasoning=args.enable_reasoning if args.enable_reasoning else False
        )
    )

    config = GenerationConfig(
        prefix_window_tokens=args.prefix_window_tokens,
        consistency_pass_enabled=not args.disable_consistency_pass,
        prompt_language=args.prompt_language,
    )
    pipeline = ChunkWiseGenerationPipeline(
        model=model,
        tokenizer=WhitespaceTokenizer(),
        config=config,
    )

    manual_plan = _load_manual_plan(args.manual_plan_path)
    if manual_plan is not None:
        logger.info(f"Using manual plan: {args.manual_plan_path}")
        result = pipeline.run(manual_plan=manual_plan)
    else:
        logger.info("Using auto-generated plan")
        result = pipeline.run(
            topic=args.topic,
            objective=args.objective,
            target_tokens=args.target_tokens,
            audience=args.audience,
            tone=args.tone,
        )

    logger.info(f"Generation finished. Final length: {len(result.final_text)} chars")
    logger.info(
        "Quality report: "
        f"missing={len(result.qc_report.coverage_missing)}, "
        f"terminology={len(result.qc_report.terminology_issues)}, "
        f"repetition={len(result.qc_report.repetition_issues)}, "
        f"drift={len(result.qc_report.drift_issues)}, "
        f"consistency_fallback={result.qc_report.consistency_pass_used_fallback}"
    )

    if args.output is None:
        print(result.final_text)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.final_text, encoding="utf-8")
    logger.info(f"Written output to: {args.output}")


if __name__ == "__main__":
    main()

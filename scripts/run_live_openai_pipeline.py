from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

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

from backends import (
    OpenAIBackendConfig,
    OpenAIRewriteModel,
)
from backends.openai import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
from pipelines import ChunkWiseRephrasePipeline, PipelineConfig
from tokenization import WhitespaceTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chunk-wise autoregressive rephrasing with a real OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/live_rephrase_input.txt"),
        help="Input text file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, output is printed to stdout.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="Rewrite for clarity while preserving all facts, numbers, and named entities.",
        help="Style instruction for rephrasing.",
    )
    parser.add_argument(
        "--prompt-language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Prompt language for instructions sent to the model.",
    )
    parser.add_argument("--chunk-size", type=int, default=80,
                        help="Chunk size limit (in tokens or chars, see --length-mode).")
    parser.add_argument("--length-mode", type=str, default="auto",
                        choices=["auto", "token", "char"],
                        help="Length calculation mode: auto (detect script), token, or char.")
    parser.add_argument("--prefix-window-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override. Same effect as setting LLM_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional API base URL override. Same effect as setting LLM_BASE_URL.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)
    logger.info(f"Reading input from: {args.input}")
    source_text = args.input.read_text(encoding="utf-8")
    logger.info(f"Input text length: {len(source_text)} chars")

    logger.info("Configuration:")
    logger.info(f"  chunk_size={args.chunk_size}")
    logger.info(f"  length_mode={args.length_mode}")
    logger.info(f"  prefix_window_tokens={args.prefix_window_tokens}")
    logger.info(f"  temperature={args.temperature}")
    logger.info(f"  top_p={args.top_p}")
    logger.info(f"  max_new_tokens={args.max_new_tokens}")
    logger.info(f"  style='{args.style}'")
    logger.info(f"  prompt_language={args.prompt_language}")

    model = OpenAIRewriteModel(
        config=OpenAIBackendConfig(
            base_url=args.base_url or DEFAULT_BASE_URL,
            model=args.model or DEFAULT_MODEL,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    )

    pipeline = ChunkWiseRephrasePipeline(
        model=model,
        tokenizer=WhitespaceTokenizer(),
        config=PipelineConfig(
            chunk_size=args.chunk_size,
            length_mode=args.length_mode,
            prefix_window_tokens=args.prefix_window_tokens,
            fidelity_threshold=0.0,
            max_retries=1,
            global_anchor_mode="head",
            prompt_language=args.prompt_language,
        ),
    )

    logger.info("Starting pipeline...")
    rewritten = pipeline.run(source_text, style_instruction=args.style)
    logger.info(f"Pipeline completed. Output length: {len(rewritten)} chars")

    if args.output is None:
        logger.info("Outputting to stdout")
        print(rewritten)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rewritten, encoding="utf-8")
        logger.info(f"Written output to: {args.output}")


if __name__ == "__main__":
    main()

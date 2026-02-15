# Chunk-wise Data Synthesis

[English](./README.md) | [简体中文](./README.zh-CN.md)

A minimal, test-covered implementation of chunk-wise autoregressive rewriting inspired by Kimi-K2:

1. Split long text into chunks.
2. Rewrite chunk `i` conditioned on generated prefix `y_<i`.
3. Stitch rewritten chunks back into one document.
4. Retry low-fidelity chunks when a verifier is enabled.

## Features

- Overlap-aware chunk splitting and stitching.
- Autoregressive rewriting with rolling prefix window.
- Global anchor control (`head` or `none`).
- Pluggable fidelity verification and retry loop.
- OpenAI-compatible backend with environment-based configuration.

## Project Layout

```text
src/
  chunking.py         # chunk split and overlap logic
  pipeline.py         # end-to-end orchestration
  prompting.py        # prompt construction
  openai_backend.py   # real API backend
  fidelity.py         # verifier interfaces and defaults
tests/
  test_*.py           # deterministic unittest coverage
scripts/
  run_live_openai_pipeline.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Tests

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run a single test module during iteration:

```bash
PYTHONPATH=src python3 -m unittest tests.test_pipeline -v
```

## Real Pipeline Run (Live API)

This is the current real end-to-end run flow:

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_pipeline.py --input tests/data/live_rephrase_input.txt --output tests/data/output.txt
```

Notes:

- `--output` is optional; if omitted, rewritten text is printed to stdout.
- Sample input file: `tests/data/live_rephrase_input.txt`.
- Example output target: `tests/data/output.txt`.

## Live Integration Test (Opt-in)

The live integration test makes a real API request and is disabled by default:

```bash
export LLM_API_KEY=your_key_here
export RUN_LIVE_LLM_TESTS=1
PYTHONPATH=src python3 -m unittest tests.test_openai_backend_live -v
```

## Configuration

Environment variables:

- `LLM_API_KEY` (required): API key.
- `LLM_MODEL` (optional): override model ID.
- `LLM_BASE_URL` (optional): override provider base URL.

Current defaults in `src/openai_backend.py`:

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"`
- `DEFAULT_MODEL = "stepfun/step-3.5-flash:free"`

You can also override from script flags:

- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`

## Minimal API Usage

```python
from pipeline import ChunkWiseRephrasePipeline, PipelineConfig
from prompting import RewriteRequest
from tokenizer import WhitespaceTokenizer


class EchoModel:
    def rewrite(self, request: RewriteRequest) -> str:
        return request.current_chunk


pipeline = ChunkWiseRephrasePipeline(
    model=EchoModel(),
    tokenizer=WhitespaceTokenizer(),
    config=PipelineConfig(
        chunk_tokens=256,
        overlap_tokens=64,
        prefix_window_tokens=1024,
        fidelity_threshold=0.85,
        max_retries=2,
    ),
)

rewritten = pipeline.run(
    "Your long document here.",
    style_instruction="Rewrite for clarity while preserving facts.",
)
print(rewritten)
```

## Troubleshooting

- Error contains `not a valid model ID`:
  set a provider-valid model, for example:
  `export LLM_MODEL=your_valid_model_id`.
- Missing API key error:
  make sure `LLM_API_KEY` is exported in the current shell.

# Chunk-wise Data Synthesis

[English](./README.md) | [简体中文](./README.zh-CN.md)

A test-covered implementation of chunk-wise long-text synthesis with two parallel pipelines inspired by Kimi-K2:

1. `ChunkWiseRephrasePipeline`: faithful chunk-wise autoregressive rephrasing.
2. `ChunkWiseGenerationPipeline`: plan-driven chunk-wise autoregressive long-form generation.

## Features

- Hierarchical no-overlap chunk splitting with overlap-aware stitching.
- Autoregressive generation with rolling prefix windows.
- Parallel workflows for rephrase and pure generation.
- Rephrase retries with pluggable fidelity verification.
- Generation section retries with issue-targeted repair prompts.
- Optional prompt compression for long-context section generation.
- Plan + state based long-form generation with consistency pass guard.
- Built-in quality checks for coverage, terminology, repetition, drift, and required entities.
- OpenAI-compatible backend with environment-based configuration.

## Project Layout

```text
src/
  chunking.py             # chunk split and overlap logic
  pipeline.py             # chunk-wise rephrase orchestration
  prompting.py            # rephrase prompt construction
  generation_pipeline.py  # chunk-wise long-form generation orchestration
  generation_prompting.py # generation prompt templates
  generation_state.py     # state table update logic
  generation_quality.py   # coverage/consistency/repetition checks
  generation_types.py     # generation dataclasses and result types
  openai_backend.py       # OpenAI-compatible backend implementations
  fidelity.py             # rephrase fidelity verifier interfaces
tests/
  test_*.py               # deterministic unittest coverage
scripts/
  run_live_openai_pipeline.py             # live rephrase runner
  run_live_openai_generation_pipeline.py  # live generation runner
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

Run one module during iteration:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_generation_pipeline.py' -v
```

## Live Rephrase Run

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_pipeline.py \
  --input tests/data/live_rephrase_input.txt \
  --output tests/data/rephrase_output.txt
```

## Live Generation Run

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_generation_pipeline.py \
  --topic "Chunk-wise autoregressive long-form generation" \
  --objective "Create long-context training text" \
  --target-tokens 1800 \
  --audience "ML engineers" \
  --tone "neutral technical" \
  --output tests/data/generation_output.txt
```

You can also pass a manual plan JSON:

```bash
python3 scripts/run_live_openai_generation_pipeline.py \
  --manual-plan-path tests/data/manual_plan.json \
  --output tests/data/generation_output.txt
```

## Live Integration Test (Opt-in)

The live integration test makes a real API request and is disabled by default:

```bash
export LLM_API_KEY=your_key_here
export RUN_LIVE_LLM_TESTS=1
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_openai_backend_live.py' -v
```

## Configuration

Environment variables:

- `LLM_API_KEY` (required): API key.
- `LLM_MODEL` (optional): override model ID.
- `LLM_BASE_URL` (optional): override provider base URL.

Current defaults in `src/openai_backend.py`:

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"`
- `DEFAULT_MODEL = "stepfun/step-3.5-flash:free"`

Live rephrase script flags (`scripts/run_live_openai_pipeline.py`):

- `--chunk-size`
- `--length-mode` (`auto` / `token` / `char`)
- `--prefix-window-tokens`
- `--style`
- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`
- `--verbose`

Live generation script flags (`scripts/run_live_openai_generation_pipeline.py`):

- `--topic`
- `--objective`
- `--target-tokens`
- `--audience`
- `--tone`
- `--manual-plan-path`
- `--prefix-window-tokens`
- `--disable-consistency-pass`
- `--enable-reasoning`
- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`
- `--verbose`

Key rephrase pipeline config (`PipelineConfig` in `src/pipeline.py`):

- `chunk_size`
- `length_mode`
- `prefix_window_tokens`
- `max_retries`
- `fidelity_threshold`
- `max_stitch_overlap_tokens`
- `global_anchor_mode`

Key generation pipeline config (`GenerationConfig` in `src/generation_types.py`):

- `prefix_window_tokens`
- `max_section_retries`
- `section_quality_threshold`
- `prompt_compression_enabled`
- `retry_on_missing_entities`
- `consistency_pass_enabled`

## Minimal API Usage

### Rephrase pipeline

```python
from pipeline import ChunkWiseRephrasePipeline, PipelineConfig
from prompting import RewriteRequest
from tokenizer import WhitespaceTokenizer


class EchoRewriteModel:
    def rewrite(self, request: RewriteRequest) -> str:
        return request.current_chunk


pipeline = ChunkWiseRephrasePipeline(
    model=EchoRewriteModel(),
    tokenizer=WhitespaceTokenizer(),
    config=PipelineConfig(
        chunk_size=256,
        length_mode="token",
        prefix_window_tokens=1024,
        max_stitch_overlap_tokens=64,
    ),
)

rewritten = pipeline.run("Your long document here.", style_instruction="Rewrite for clarity.")
print(rewritten)
```

### Generation pipeline (manual plan)

```python
from generation_pipeline import ChunkWiseGenerationPipeline
from generation_types import GenerationConfig, GenerationPlan, SectionSpec
from model import LLMRequest
from tokenizer import WhitespaceTokenizer


class StubLLM:
    def generate(self, request: LLMRequest) -> str:
        if request.task == "section_generation":
            return "Section body with required entities and key points."
        if request.task == "consistency_pass":
            return "Section body with required entities and key points."
        raise ValueError("manual plan run should not call plan_generation")


plan = GenerationPlan(
    topic="Chunk-wise generation",
    objective="Teach the method",
    audience="ML engineers",
    tone="neutral technical",
    target_total_length=300,
    sections=[
        SectionSpec(
            title="Intro",
            key_points=["global anchor controls structure"],
            required_entities=["global anchor"],
            constraints=[],
            target_length=120,
        )
    ],
    terminology_preferences={"global anchor": "global anchor"},
    narrative_voice="third-person",
    do_not_include=[],
)

pipeline = ChunkWiseGenerationPipeline(
    model=StubLLM(),
    tokenizer=WhitespaceTokenizer(),
    config=GenerationConfig(prefix_window_tokens=800),
)

result = pipeline.run(manual_plan=plan)
print(result.final_text)
print(result.qc_report.coverage_missing)
```

## Troubleshooting

- Error contains `not a valid model ID`:
  set a provider-valid model, for example:
  `export LLM_MODEL=your_valid_model_id`.
- Missing API key error:
  make sure `LLM_API_KEY` is exported in the current shell.

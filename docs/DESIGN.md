# DESIGN

Canonical architecture reference for this repository.

## Scope

This project provides chunk-wise autoregressive text synthesis with two pipelines:
- Rephrase pipeline for fidelity-preserving rewriting.
- Generation pipeline for long-form sectioned drafting.

## Layered Module Boundaries

Use the package layout under `src/` as the source of truth:

- Pipeline orchestration: `pipelines/rephrase.py`, `pipelines/generation.py`, `pipelines/base.py`
- Prompt rendering: `prompts/rephrase.py`, `prompts/generation.py`, `prompts/base.py`
- Quality checks: `quality/fidelity.py`, `quality/generation.py`, `quality/base.py`
- Backend adapters: `backends/openai.py`
- Core contracts/types/config: `core/protocols.py`, `core/types.py`, `core/config.py`
- Domain modules: `chunking.py`, `generation_state.py`, `generation_types.py`, `model.py`

## Architecture Contracts

- Keep responsibilities inside the owning layer; avoid cross-layer deep imports.
- Public APIs should be exposed through package entrypoints.
- Do not re-introduce removed legacy wrappers (`pipeline.py`, `prompting.py`, `fidelity.py`, `openai_backend.py`).
- For behavior changes, update implementation and docs in one change set.

## Design Drill-Down

- Deep technical designs live in [`docs/design-docs/`](design-docs/README.md).
- Execution lifecycle plans live in [`docs/exec-plans/`](exec-plans/README.md).
- Product intent and tradeoffs live in [`docs/PRODUCT_SENSE.md`](PRODUCT_SENSE.md).

# AGENTS Map

This file is a concise repository map. Canonical detail lives under `docs/`.

## Start Here

- Architecture and module boundaries: [`docs/DESIGN.md`](docs/DESIGN.md)
- Plan lifecycle and index: [`docs/PLANS.md`](docs/PLANS.md)
- Product intent and quality bars: [`docs/PRODUCT_SENSE.md`](docs/PRODUCT_SENSE.md)
- Milestone direction: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Frontend scope statement: [`docs/FRONTEND.md`](docs/FRONTEND.md)

## Repository Map

- Source: `src/`
- Tests: `tests/`
- Runtime scripts: `scripts/`
- Context system: `docs/`

## Source Layout (High-Level)

- Pipelines: `src/pipelines/`
- Prompts: `src/prompts/`
- Quality checks: `src/quality/`
- Backend adapters: `src/backends/`
- Core contracts and config: `src/core/`
- Domain modules: `src/chunking.py`, `src/generation_state.py`, `src/generation_types.py`, `src/model.py`

## Testing and Validation

- Full offline unit tests:
  - `uv run python -m unittest discover -s tests -v`
- Focused module test:
  - `uv run python -m unittest tests.test_generation_pipeline -v`
- Docs structure and link validation:
  - `uv run python scripts/validate_docs.py`

## Live Runs (Opt-In)

- Rephrase flow:
  - `uv run python scripts/run_live_openai_pipeline.py --input tests/data/live_rephrase_input.txt`
- Generation flow:
  - `uv run python scripts/run_live_openai_generation_pipeline.py --topic "..." --objective "..."`
- Optional live backend test:
  - `RUN_LIVE_LLM_TESTS=1 LLM_API_KEY=... uv run python -m unittest tests.test_openai_backend_live -v`

## Context Governance Rules

- Keep `AGENTS.md` as a map (no deep specifications).
- Store durable knowledge under `docs/` as single source of truth.
- Treat execution plans as first-class artifacts in `docs/exec-plans/`:
  - `active/`
  - `completed/`
  - `tech-debt/`
- When plan state changes, update both file location and `docs/PLANS.md` index.
- Keep legacy `docs/plans/` paths only as migration redirects.

## Docs Update Policy

For architecture/workflow changes in code:

1. Update implementation.
2. Update the relevant canonical docs in `docs/`.
3. Run docs validator and unit tests.
4. Include docs changes in the same PR/commit.

## Safety and Configuration

- Never commit secrets or API keys.
- Use environment variables:
  - `LLM_API_KEY` (required for live calls)
  - `LLM_MODEL` (optional)
  - `LLM_BASE_URL` (optional)

## Legacy Note

`docs/plans/` is preserved as a redirect layer for compatibility. Do not add new canonical content there.

# 2026-04-11 Docs Validator CI Gate

## Status

Open tech debt.

## Problem

Docs validation runs locally but is not yet guaranteed by all CI workflows.

## Impact

Without CI enforcement, regressions in docs structure or links can merge unnoticed.

## Resolution Path

- Add `python scripts/validate_docs.py` to CI checks.
- Ensure failure blocks merge.
- Keep `tests/test_docs_validation.py` in default unit suite.

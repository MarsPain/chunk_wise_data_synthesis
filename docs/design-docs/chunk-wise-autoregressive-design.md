# Chunk-wise Autoregressive Rephrasing Design

## Goal

Implement a minimal but correct chunk-wise autoregressive rephrasing flow for long documents, with extension points for stronger fidelity verification.

## Alternatives Considered

1. Flat token chunking only.
- Pros: simplest implementation.
- Cons: chunk boundaries can cut discourse units and reduce readability.

2. Two-pass structural plus token chunking. (Chosen)
- Pros: preserves paragraph-level boundaries first, then token chunking when needed.
- Cons: still heuristic; no syntax-aware sentence segmentation.

3. Full document rewrite with a long-context model.
- Pros: simplest prompt shape.
- Cons: higher cost and still exposed to output-length truncation.

## Chosen Architecture

- `chunking.py`: structure-first splitting, then token chunking with overlap.
- `pipelines/rephrase.py`: sequential autoregressive rewrite loop.
- `quality/fidelity.py`: pluggable verifier (`NoOpVerifier`, lexical baseline).
- `prompts/rephrase.py`: request contract between pipeline and backend.
- `pipelines/base.py`: shared stitching and pipeline helpers.

## Validation Strategy

- Unit tests for overlap chunking behavior.
- Unit tests for autoregressive prefix propagation.
- Unit tests for fidelity-failure retry behavior.
- Unit tests for overlap-aware stitching de-duplication.

## Notes

This document was migrated from `docs/plans/2026-02-14-chunk-wise-autoregressive-design.md` to align with the docs information architecture.

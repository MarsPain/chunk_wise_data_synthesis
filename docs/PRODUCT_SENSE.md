# PRODUCT_SENSE

## Users

- Developers integrating chunk-wise rewriting or generation flows.
- Researchers iterating on prompt, fidelity, and orchestration strategies.

## User Value

- Reliable long-input handling through chunked workflows.
- Deterministic, testable behavior with stubs/fakes for local development.
- Clear extension points for prompt strategy and quality verification.

## Quality Bars

- Correctness: no silent data loss at chunk boundaries.
- Reliability: retries and guardrails for malformed/low-fidelity outputs.
- Operability: offline unit tests remain fast and reproducible.

## Non-Goals

- Full no-code product UI.
- Hard dependency on live network calls for local validation.

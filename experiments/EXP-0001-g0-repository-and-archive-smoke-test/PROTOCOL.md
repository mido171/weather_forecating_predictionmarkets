# Predeclared Protocol

Complete before inspecting validation/locked-test outcomes.

## Target and horizon

- target version: not used in G0
- rules/target adapter version: not used in G0
- horizon: not selected; G2 remains open
- exact cutoff: not applicable
- prediction unit: no predictions are generated

## Sample

- development: not applicable
- validation: repository self-tests and config/source validation
- locked test: not opened
- live shadow: not started
- inclusion: all sources tagged `bootstrap_now` in `config/data_sources.yaml`
- exclusion: all non-bootstrap sources and all modelling work
- expected row count: 7 sources, fetched twice

## Baseline

- champion/baseline version: none
- frozen prediction artifact: none
- reason: G0 validates infrastructure before any baseline exists

## Candidate

- feature/formula/model: raw archive smoke test
- transformations: none
- allowed hyperparameters: none
- selection procedure: fetch every configured `bootstrap_now` source exactly as
  configured
- seeds: none used
- compute budget: local dependency setup, two source-fetch passes, validation,
  and focused archive verification

## Metrics

- primary: binary G0 acceptance pass/fail
- guardrails: no failed doctor/test/validation checks; all raw hashes
  independently recomputable; no missing sidecars or HTTP metadata
- calibration: not applicable
- subgroup: per-source archive verification
- operational: repeated retrievals are traceable and raw archive is immutable

## Uncertainty

- method: deterministic command exit status and byte/hash verification
- block length: not applicable
- repetitions: two live fetch passes
- confidence level: not applicable

## Multiplicity

- experiment family: G0 infrastructure smoke test
- number of variants: one
- correction/confirmation approach: rerun checks after any patch

## Acceptance

- `doctor` passes.
- `pytest` passes.
- `validate all` passes, allowing only expected governance warnings for G1/G2.
- `ruff check src tests scripts` and `mypy src` pass after code/test edits.
- Every `bootstrap_now` source has at least two retrieval events.
- Every retrieval has non-empty raw bytes, SHA-256, retrieval timestamp, HTTP
  metadata, and metadata sidecar.
- Raw hashes recompute exactly.
- Failure paths for HTTP error, empty payload, and malformed HKO CSV are tested.
- `.env` exists locally and no secret file is committed.

## Rejection

Reject or block G0 if any acceptance command fails after a proper fix attempt,
if any bootstrap source cannot be archived, if raw archive integrity cannot be
verified, or if the repository would commit secrets.

## Locked-test decision

Not authorized. G0 does not access modelling data splits or locked-test
outcomes.

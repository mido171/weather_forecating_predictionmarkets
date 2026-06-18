# EXP-0001 — G0 repository and archive smoke test

**Status:** ACCEPTED  
**Created:** 2026-06-18T16:08:31.408079Z  
**Goal dependency:** G0  
**Owner:** Codex

## Plain-language question

Can this repository bootstrap, validate itself, fetch every `bootstrap_now`
source twice, and prove that raw source snapshots are immutable,
hashed, timestamped, and documented with HTTP metadata?

## Why this could improve Tmax

This does not improve a Tmax forecast directly. It proves the archive and
validation foundation required before target parity, baselines, or modelling
can be trusted.

## What changed from prior work

This is the first experiment in the repository. During the smoke test, the
PowerShell bootstrap script was patched so native command failures stop the
script instead of printing a false success message.

## Current conclusion

ACCEPTED for G0. The repository passed doctor, tests, validation, lint, type
checking, source fetching, archive verification, and source inventory
generation. Predictive modelling remains blocked by G1/G2.

## Navigation

- hypothesis: `HYPOTHESIS.md`
- protocol: `PROTOCOL.md`
- as-of contract: `ASOF_CONTRACT.md`
- data: `DATA_MANIFEST.yaml`
- configuration: `RUN_CONFIG.yaml`
- results: `RESULTS.md`
- conclusion: `CONCLUSION.md`
- reproduction: `REPRODUCE.md`
- gates: `STATUS.yaml`

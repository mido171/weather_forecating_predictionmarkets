# Conclusion

## Decision

`ACCEPTED`

Choose exactly one:

- ACCEPTED
- REJECTED
- INCONCLUSIVE
- BLOCKED

## What the evidence supports

The G0 repository and archive smoke test passed after proper dependency setup
and a patch to make `scripts/bootstrap.ps1` fail loudly on native command
errors. All `bootstrap_now` sources were archived twice with raw bytes,
sidecars, SHA-256 hashes, retrieval timestamps, request/response metadata,
2xx HTTP status, and response headers. Independent hash verification passed.

## What the evidence does not support

This experiment does not establish settlement target parity, CLMMAXT parity,
provider publication timing, primary forecast horizon, source-specific latency,
baseline skill, calibration, market value, or any predictive model.

## Improvement over baseline

No forecast baseline exists. The relevant metric is binary G0 acceptance:
PASS for 7 bootstrap sources and 14 retrieval events.

## Mechanism assessment

Supported for archive mechanics and repository validation. Not a meteorological
mechanism test.

## Robustness

The fetch was run twice. The same hashes were observed for unchanged live
payloads, but each retrieval produced a distinct timestamped immutable path.
Failure-path tests cover HTTP-error, empty-payload, and malformed CSV cases.

## Leakage review

PASS for G0. No predictive features, labels, train/test splits, locked-test
outcomes, reanalysis, best tracks, or target-day observations were used.
Raw source retrieval timestamps were preserved and no eligibility beyond raw
archive availability was claimed.

## Reproducibility review

PASS for G0 local reproducibility. Doctor, tests, validation, ruff, mypy,
manifest generation, source inventory generation, and archive verification
passed from the local `.venv`. The PowerShell bootstrap script now fails
loudly if native commands fail.

## Operational viability

PASS for G0 archive smoke operation. Live archival beyond bootstrap sources,
continuous monitors, source contracts, and parsing are still later-goal work.

## Milestone eligibility

- [ ] material OOS improvement
- [x] leakage PASS
- [x] reproducibility PASS
- [ ] calibration/tail guardrails PASS
- [x] operationally available
- [ ] eligible for MILESTONES

Not eligible for `MILESTONES.md` as a model or forecast improvement because
there is no out-of-sample predictive result.

## New hypotheses generated

- G1 must prove whether first-published Daily Extract `Absolute Daily Max
  (deg. C)` and `CLMMAXT station=HKO` match date by date.
- Source-specific parsers should extract provider issue/update timestamps for
  the live HKO forecast and latest-observation feeds.

## Final next action

Start G1 only: create an experiment for Daily Extract versus CLMMAXT target
parity and archive/parse the settlement-relevant HKO and Polymarket source
evidence before any modelling.

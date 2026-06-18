# Hypothesis

## Mechanism

First-publication evidence requires repeated polling around expected HKO Daily
Extract publication. A safe poller must support bounded repeat runs, but should
not promote every newly archived row to provider-first-publication candidate
status. Candidate labels should require both an active polling start timestamp
and an explicit local-date allowlist.

## Exact Prediction

- `scripts/poll_daily_extract.py` can run multiple poll iterations with a
  configured interval and then stop.
- The ledger can accept an active polling start timestamp and watched dates.
- A row is labelled `PROVIDER_FIRST_PUBLICATION_CANDIDATE` only if:
  - its first archived observation is at or after active polling start; and
  - its local date is explicitly listed as a watched candidate date; and
  - no revision is observed.
- Unwatched historical rows remain `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST`.
- Revisions still override candidate status as `REVISION_OBSERVED`.

## Null Hypothesis

The poller either cannot run repeated bounded polling safely, or it can
mislabel historical/unwatched rows as provider-first-publication candidates.

## Falsification

- A repeated polling run leaves an uncontrolled background process.
- An unwatched date receives `PROVIDER_FIRST_PUBLICATION_CANDIDATE`.
- A watched date observed before active polling start receives candidate status.
- A revised date receives candidate status.
- Tests, validation, Ruff, or mypy fail.

## Leakage Risks Anticipated

Candidate labels are source-timing evidence, not model features. The default
must remain conservative, and G1 cannot pass until a candidate row is reviewed
against polling cadence and source behavior.

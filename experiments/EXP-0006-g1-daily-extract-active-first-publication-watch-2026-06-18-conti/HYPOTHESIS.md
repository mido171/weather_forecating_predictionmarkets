# Hypothesis

## Claim

If the HKO Daily Extract `2026-06-18` row appears after the active watch began
at `2026-06-18T17:48:59.956593Z`, the ledger can identify it as a
provider-first-publication candidate only when an earlier active absent snapshot
and a later first-present snapshot are both available.

## Mechanism

The monthly Daily Extract backing payload is archived immutably on each poll.
EXP-0005 already archived multiple active absent snapshots. Continuing the same
watch may capture the absent-to-present transition.

## Expected Direction

Either the watched date remains missing with zero provider-first candidates, or
one `2026-06-18` candidate appears with a recorded last active absent snapshot.

## Falsification

The evidence fails if the poller cannot complete, if per-iteration raw snapshot
metadata is missing, if a candidate appears without active absent evidence, or
if later snapshots revise the first-present value.

## Leakage Risks

- Treating a latest payload as historical first publication.
- Ignoring gaps in the active watch cadence.
- Treating target-publication evidence as a model feature.
- Inferring provider publication time more precisely than the raw absent/present
  snapshots support.
